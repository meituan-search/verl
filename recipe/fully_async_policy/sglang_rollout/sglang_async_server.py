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
from typing import Any, Optional, Sequence

import ray
import torch
from ray.actor import ActorHandle
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLangHttpServerBase,
    SGLangReplica,
)
from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SGLangHttpServerForPartialBase(SGLangHttpServerBase):
    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        super().__init__(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            workers=workers,
            replica_rank=replica_rank,
            node_rank=node_rank,
            nnodes=nnodes,
            cuda_visible_devices=cuda_visible_devices,
        )

        # for cancel LLMServer
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.req_output: dict[str, Optional[dict[str, Any]]] = {}

    async def _generate_step(
        self,
        prompt_ids: list[int] | torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        """Generate tokens using SGLang tokenizer_manager.
        
        Args:
            prompt_ids: List of token IDs for the prompt
            sampling_params: Sampling parameters dict
            request_id: Unique request ID
            image_data: Optional image data for multimodal generation
            
        Note:
            This method collects all outputs from the async generator and stores
            the final output in self.req_output[request_id].
        """
        try:
            # Check if cancelled before starting
            if request_id in self.cancel_event and self.cancel_event[request_id].is_set():
                return
            
            # Prepare sampling params
            max_new_tokens = min(
                self.config.response_length,
                self.config.max_model_len - len(prompt_ids) - 1
            )
            
            # Create a copy to avoid modifying the original
            sglang_sampling_params = sampling_params.copy()
            sglang_sampling_params["max_new_tokens"] = max_new_tokens
            return_logprob = sglang_sampling_params.pop("logprobs", False)
            
            # Convert to SGLang SamplingParams
            sglang_sp = SamplingParams(
                max_new_tokens=max_new_tokens,
                return_logprob=return_logprob,
                **{k: v for k, v in sglang_sampling_params.items() if k != "max_new_tokens"}
            )
            
            # Convert prompt_ids to torch.Tensor if needed (SGLang expects list[int] or tensor)
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids_list = prompt_ids.tolist()
            else:
                prompt_ids_list = list(prompt_ids)
            
            # Create GenerateReqInput
            request = GenerateReqInput(
                rid=request_id,
                input_ids=prompt_ids_list,
                sampling_params=sglang_sp,
                return_logprob=return_logprob,
                image_data=image_data,
            )
            
            # Generate using tokenizer_manager
            generator = self.tokenizer_manager.generate_request(request, None)
            
            # Collect all outputs from the async generator
            final_output = None
            async for output in generator:
                # Check for cancellation during generation
                if request_id in self.cancel_event and self.cancel_event[request_id].is_set():
                    # Abort the request if cancelled
                    await self.tokenizer_manager.abort_request(rid=request_id, abort_all=False)
                    raise asyncio.CancelledError(f"Request {request_id} was cancelled")
                
                final_output = output
            
            # Store the final output
            if final_output is not None:
                self.req_output[request_id] = final_output
            else:
                logger.warning(f"Request {request_id} generated no output")
                self.req_output[request_id] = None
                
        except asyncio.CancelledError:
            # Request was cancelled, clean up
            if request_id in self.req_output:
                self.req_output[request_id] = None
            raise
        except Exception as e:
            logger.error(f"Error in _generate_step for request {request_id}: {e}", exc_info=True)
            self.req_output[request_id] = None
            raise

    async def generate_for_partial(
        self,
        prompt_ids: list[int] | torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> tuple[Sequence[int], list[float], bool]:
        """Generate tokens for partial rollout with cancellation support.
        
        Args:
            prompt_ids: List of token IDs for the prompt
            sampling_params: Sampling parameters dict
            request_id: Unique request ID
            image_data: Optional image data for multimodal generation
            
        Returns:
            Tuple of (token_ids, log_probs, is_cancel):
            - token_ids: Generated token IDs
            - log_probs: Log probabilities for each token
            - is_cancel: Whether the request was cancelled
        """
        async with self.lock:
            if self.paused:
                # After cancel, all tasks will return directly and wait for the next submission
                return [], [], True
            self.req_output[request_id] = None
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_ids, sampling_params, request_id, image_data)
            )

        done, pend = await asyncio.wait([generation_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)

        # Wait for completed tasks
        for task in done:
            try:
                await task
            except asyncio.CancelledError:
                logger.debug(f"Task for request {request_id} was cancelled")

        # Cancel pending tasks
        for task in pend:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self.lock:
            # Check if generation was cancelled or failed
            if request_id not in self.req_output or self.req_output[request_id] is None:
                is_cancel = generation_handle not in done or generation_handle.cancelled()
                self.cancel_event.pop(request_id, None)
                self.req_output.pop(request_id, None)
                return [], [], is_cancel
            
            # Extract output from SGLang format (dict, not RequestOutput)
            output = self.req_output[request_id]
            
            # Extract token_ids
            token_ids = output.get("output_ids", [])
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            elif not isinstance(token_ids, list):
                token_ids = list(token_ids)
            
            # Extract log_probs
            log_probs: list[float] = []
            if "meta_info" in output and "output_token_logprobs" in output["meta_info"]:
                output_token_logprobs = output["meta_info"]["output_token_logprobs"]
                # SGLang format: [(log_prob, token_ids, ...), ...]
                for log_prob, token_id, _ in output_token_logprobs:
                    log_probs.append(float(log_prob))
            else:
                # If no logprobs, create empty list
                log_probs = [0.0] * len(token_ids)
            
            is_cancel = generation_handle not in done or generation_handle.cancelled()
            
            # Clean up
            self.cancel_event.pop(request_id, None)
            self.req_output.pop(request_id, None)
            
        return token_ids, log_probs, is_cancel

    async def cancel(self):
        async with self.lock:
            self.paused = True
            for request_id in self.cancel_event:
                self.cancel_event[request_id].set()

    async def resume(self):
        async with self.lock:
            self.paused = False

    async def reset_prefix_cache(self):
        async with self.lock:
            print("Reset prefix cache ...")
            await self.tokenizer_manager.flush_cache()


@ray.remote(num_cpus=1)
class SGLangHttpServerForPartial(SGLangHttpServerForPartialBase):
    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        super().__init__(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            workers=workers,
            replica_rank=replica_rank,
            node_rank=node_rank,
            nnodes=nnodes,
            cuda_visible_devices=cuda_visible_devices,
        )


class FullyAsyncSGLangReplica(SGLangReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = SGLangHttpServerForPartial

    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])

    async def reset_prefix_cache(self):
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.reset_prefix_cache.remote() for server in self.servers])
