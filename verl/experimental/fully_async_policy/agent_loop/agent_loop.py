# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    AsyncLLMServerManager,
    TokenOutput,
)
from verl.experimental.teacher_loop import TeacherModelManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import (
    rollout_trace_op,
)
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    """FullyAsyncLLMServerManager supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        old_log_prob_server_handle: ray.actor.ActorHandle = None,
        tokenizer: Any = None,
    ):
        super().__init__(
            config=config,
            servers=servers,
            load_balancer_handle=load_balancer_handle,
            old_log_prob_server_handle=old_log_prob_server_handle,
        )
        self.old_log_prob_server_handle = old_log_prob_server_handle
        self.tokenizer = tokenizer

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None
        validate = sampling_params.pop("validate", None)

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            old_log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
            current_prompt_ids = prompt_ids + final_output.token_ids
            current_temperature = sampling_params.get("temperature", 1.0)
            # Skip old log prob computation during validation
            if not validate:
                output = await self._recompute_old_log_prob(output, current_prompt_ids, current_temperature)
            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            if output.old_log_probs is not None:
                final_output.old_log_probs.extend(output.old_log_probs)
            if output.routed_experts is not None:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat([final_output.routed_experts, output.routed_experts], dim=0)
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break
        if validate:  # restore for multi-turn use
            sampling_params["validate"] = validate
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output

    async def _recompute_old_log_prob(self, output: TokenOutput, context_prompt_ids, temperature: float):
        if self.old_log_prob_server_handle is None:
            return output
        # Convert TokenOutput -> fixed-shape TensorDict for OldLogProbServer.
        if self.config.get("actor_rollout_ref"):
            rollout_config = self.config.actor_rollout_ref.rollout
        else:
            rollout_config = self.config.rollout

        # Prompt grows during partial-rollout/multi-turn. Reserve prompt slots for
        # [original prompt + at most full response length] to keep shapes static.
        max_prompt_len = int(rollout_config.prompt_length) + int(rollout_config.response_length)
        max_response_len = int(rollout_config.response_length)

        # Only recompute old_log_probs for newly generated tokens in this turn.
        if len(output.token_ids) == 0:
            output.old_log_probs = []
            return output

        prompt_len = len(context_prompt_ids)
        response_len = len(output.token_ids)

        if prompt_len > max_prompt_len:
            raise ValueError(
                f"prompt length {prompt_len} exceeds padded prompt length {max_prompt_len} "
                "for old_log_prob recomputation"
            )
        if response_len > max_response_len:
            print(
                f"response length {response_len} exceeds padded response length {max_response_len} "
                "for old_log_prob recomputation"
            )
            output.token_ids = output.token_ids[:max_response_len]
            response_len = max_response_len

        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("tokenizer is required for old_log_prob recomputation padding")

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        prompt_output = tokenizer.pad(
            {"input_ids": context_prompt_ids},
            padding="max_length",
            max_length=max_prompt_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        tokenizer.padding_side = "right"
        response_output = tokenizer.pad(
            {"input_ids": output.token_ids},
            padding="max_length",
            max_length=max_response_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)
        tokenizer.padding_side = original_padding_side

        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)
        response_mask = response_output["attention_mask"]

        data_td = TensorDict(
            {
                "prompts": prompt_output["input_ids"],
                "responses": response_output["input_ids"],
                "input_ids": input_ids,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "attention_mask": attention_mask,
                "loss_mask": response_mask,
            },
            batch_size=[1],
        )

        tu.assign_non_tensor(
            data_td,
            temperature=temperature,
            max_response_len=max_response_len,
        )
        result_td = await self.old_log_prob_server_handle.compute_old_log_prob.remote(data_td)

        # Keep only valid response tokens; drop right-padding region.
        log_probs_tensor = tu.get(result_td, "log_probs")
        output.old_log_probs = log_probs_tensor[0, :response_len].tolist()
        return output


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorker):
    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        teacher_servers: list[tuple[str, ray.actor.ActorHandle]] = None,
        teacher_load_balancer_handle: ray.actor.ActorHandle = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        old_log_prob_server_handle: ray.actor.ActorHandle = None,
    ):
        super().__init__(
            config,
            servers,
            load_balancer_handle,
            teacher_servers,
            teacher_load_balancer_handle,
            reward_loop_worker_handles,
        )
        self.server_manager = FullyAsyncLLMServerManager(
            config,
            servers,
            load_balancer_handle,
            old_log_prob_server_handle,
            tokenizer=self.tokenizer,
        )


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        teacher_model_manager: TeacherModelManager = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        old_log_prob_server_handle: ray.actor.ActorHandle = None,
    ):
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        super().__init__(
            config,
            worker_group,
            rollout_resource_pool,
            teacher_model_manager,
            reward_loop_worker_handles,
            old_log_prob_server_handle,
        )
        if self.distillation_enabled:
            raise NotImplementedError("Distillation is not implemented in FullyAsyncAgentLoopManager yet.")

    @auto_await
    async def generate_sequences_single(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch. Single sample data
        Returns:
            DataProto: Output batch.
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(prompts)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker
