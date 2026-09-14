# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
            # Piggyback gate, mirrored from AgentLoopWorker.generate_sequences
            # (agent_loop.py). Without this key the rollout client pops a
            # default False (llm_server.py), resumed trajectories never emit
            # prefix_prompt_logprobs, and every multi-segment trajectory lands
            # in case 2 with reason "no_resume_version".
            enable_piggyback=getattr(config, "enable_piggyback", False),
            # token_versions gate: only partial_reprefill's token-level
            # staleness diagnostics consume the field; other trainers skip
            # the per-trajectory construction cost entirely.
            emit_token_versions=getattr(config, "emit_token_versions", False),
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            # Promote partial_rollout piggyback fields (written by the llm_server
            # client into extra_fields as lists) to top-level tensor fields so
            # they travel with the main put.
            extra = output.extra_fields or {}
            if "token_versions" in extra:
                field["token_versions"] = torch.tensor(extra["token_versions"], dtype=torch.int32)
            if "new_rollout_log_probs" in extra:
                field["new_rollout_log_probs"] = torch.tensor(extra["new_rollout_log_probs"], dtype=torch.float32)
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tag = {
                "status": "success",
                "prompt_len": prompt_len,
                "response_len": response_len,
                "seq_len": prompt_len + response_len,
                # These tags are used for off-policy staleness control, if a trajectory
                # spans too many global steps, we need to filter it out.
                # global_steps: which global steps this sample is from dataloader
                "global_steps": kwargs["global_steps"],
                # min_global_steps: start generation model weights version of this trajectory
                "min_global_steps": field["extra_fields"].get("min_global_steps"),
                # max_global_steps: end generation model weights version of this trajectory
                "max_global_steps": field["extra_fields"].get("max_global_steps"),
            }
            # partial_rollout piggyback scalars travel as per-trajectory tag entries.
            if field["extra_fields"].get("piggyback_marker"):
                tag["piggyback_marker"] = True
            if "resume_version" in field["extra_fields"]:
                tag["resume_version"] = int(field["extra_fields"]["resume_version"])
            # Record per-trajectory performance metrics in the tag so they
            # sync to the replay buffer for free (fields need an explicit
            # kv_batch_get) and the trainer can aggregate agent_loop/* timing
            # from the sampled batch's tags.
            m = output.metrics
            tag["agent_loop_metrics"] = {
                "generate_sequences": float(m.generate_sequences),
                "tool_calls": float(m.tool_calls),
                "compute_score": float(m.compute_score),
                "num_preempted": int(m.num_preempted),
            }
            tags.append(tag)

        partition_id = "train" if not validate else "val"
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )

    def _performance_metrics(self, tags: list) -> dict:
        """Aggregate per-trajectory agent_loop metrics from the sampled batch's tags.

        TQ-path mirror of AgentLoopManager._performance_metrics: the base
        class aggregates per-sample metrics from DataProto outputs returned
        by generate_sequences; here workers record the same AgentLoopMetrics
        values into each trajectory's TQ tag (see _agent_loop_postprocess),
        and the trainer calls this with the sampled batch's tags. Emits the
        same agent_loop/* keys, plus the slowest trajectory's prompt/response
        lengths from its tag.
        """
        rows = [(tag, tag["agent_loop_metrics"]) for tag in tags if tag and tag.get("agent_loop_metrics")]
        if not rows:
            return {}
        t_generate_sequences = np.array([float(r["generate_sequences"]) for _, r in rows])
        t_tool_calls = np.array([float(r["tool_calls"]) for _, r in rows])
        t_compute_score = np.array([float(r["compute_score"]) for _, r in rows])
        num_preempted = np.array([int(r["num_preempted"]) for _, r in rows])

        timing = {
            "agent_loop/num_preempted/min": num_preempted.min(),
            "agent_loop/num_preempted/max": num_preempted.max(),
            "agent_loop/num_preempted/mean": num_preempted.mean(),
            "agent_loop/generate_sequences/min": t_generate_sequences.min(),
            "agent_loop/generate_sequences/max": t_generate_sequences.max(),
            "agent_loop/generate_sequences/mean": t_generate_sequences.mean(),
            "agent_loop/tool_calls/min": t_tool_calls.min(),
            "agent_loop/tool_calls/max": t_tool_calls.max(),
            "agent_loop/tool_calls/mean": t_tool_calls.mean(),
            "agent_loop/compute_score/min": t_compute_score.min(),
            "agent_loop/compute_score/max": t_compute_score.max(),
            "agent_loop/compute_score/mean": t_compute_score.mean(),
        }

        # batch sequence generation is bounded by the slowest sample
        slowest = int(np.argmax(t_generate_sequences + t_tool_calls + t_compute_score))
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]
        timing["agent_loop/slowest/prompt_length"] = float(rows[slowest][0].get("prompt_len", 0))
        timing["agent_loop/slowest/response_length"] = float(rows[slowest][0].get("response_len", 0))
        return timing
