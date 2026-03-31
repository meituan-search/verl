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

import ray
import torch
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.protocol import DataProtoFuture
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.utils.profiler.performance import log_gpu_memory_usage
from verl.workers.config import HFModelConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker, _with_routing_replay_flag
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class OldLogProbWorker(Worker, DistProfilerExtension):
    """Worker for computing old log probabilities."""

    def __init__(self, config: DictConfig, role: str = "old_log_prob"):
        Worker.__init__(self)
        self.config = config
        self.role = role

        self.old_log_prob: TrainingWorker = None
        self.training_worker_config: TrainingWorkerConfig = None
        self.enable_routing_replay = False
        self._staged_state_dict = None

        omega_profiler_config = self.config.old_log_prob.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        self.enable_routing_replay = (
            self.config.old_log_prob.strategy == "megatron"
            and self.config.old_log_prob.megatron.router_replay.mode != "disabled"
        )

        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize internal TrainingWorker and register dispatch info."""

        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
        with open_dict(self.config.old_log_prob):
            self.config.old_log_prob.pop("enable_standalone", None)
            self.config.old_log_prob.pop("nnodes", None)
            self.config.old_log_prob.pop("n_gpus_per_node", None)

            self.config.old_log_prob.ppo_mini_batch_size = self.config.old_log_prob.micro_batch_size_per_gpu
            self.config.old_log_prob.ppo_micro_batch_size_per_gpu = self.config.old_log_prob.pop(
                "micro_batch_size_per_gpu", None
            )
            self.config.old_log_prob.ppo_max_token_len_per_gpu = self.config.old_log_prob.pop(
                "max_token_len_per_gpu", None
            )

        old_log_prob_config = omega_conf_to_dataclass(self.config.old_log_prob)
        old_log_prob_config.model_config = model_config

        # construct TrainingWorkerConfig
        training_worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=old_log_prob_config.model_config,
            engine_config=old_log_prob_config.engine,
        )

        # assign engine configs
        training_worker_config.engine_config.use_dynamic_bsz = self.config.old_log_prob.use_dynamic_bsz
        training_worker_config.engine_config.infer_max_token_len_per_gpu = (
            self.config.old_log_prob.ppo_max_token_len_per_gpu,
        )
        training_worker_config.engine_config.infer_micro_batch_size_per_gpu = (
            self.config.old_log_prob.ppo_micro_batch_size_per_gpu
        )
        training_worker_config.engine_config.max_token_len_per_gpu = self.config.old_log_prob.ppo_max_token_len_per_gpu
        training_worker_config.engine_config.micro_batch_size_per_gpu = (
            self.config.old_log_prob.ppo_micro_batch_size_per_gpu
        )
        training_worker_config.engine_config.use_remove_padding = model_config.use_remove_padding
        if self.config.old_log_prob.use_dynamic_bsz:
            assert self.config.old_log_prob.ppo_max_token_len_per_gpu is not None
        else:
            assert self.config.old_log_prob.ppo_micro_batch_size_per_gpu is not None

        self.old_log_prob = TrainingWorker(config=training_worker_config)
        self.old_log_prob.reset()
        log_gpu_memory_usage("[Old_log_prob] After init model", logger=logger)
        self.set_dispatch_collect(mesh_name="old_log_prob", **self.old_log_prob.get_dispatch_collect())

        # Build checkpoint engine (as receiver, is_master=False)
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            backend, is_master=False, bucket_size=bucket_size, **engine_kwargs
        )

        # Free cached GPU memory
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """
        Receive weights from trainer via checkpoint engine and stage to CPU.
        """
        if self.checkpoint_engine is None:
            return
        # Receive weights and stage to CPU
        staged = {}
        async for name, tensor in self.checkpoint_engine.receive_weights():
            staged[name] = tensor.clone().to("cpu", non_blocking=True)
            logger.debug(f"[OldLogProbWorker] update_weights, {name=}, shape={tensor.shape}")

        torch.cuda.synchronize()
        self._staged_state_dict = staged
        logger.info(f"[OldLogProbWorker] Rank {self.rank}: staged {len(staged)} params to CPU")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_staged_weights(self):
        """
        Load staged CPU weights into the model engine.
        """
        if self._staged_state_dict is None or self.old_log_prob is None:
            return
        self.old_log_prob.engine.set_param(self._staged_state_dict)

        self._staged_state_dict = None
        torch.cuda.empty_cache()
        logger.info(f"[OldLogProbWorker] Rank {self.rank}: loaded staged weights into engine")

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Proxy method for CheckpointEngineManager to call prepare/init_process_group/finalize."""
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="old_log_prob"), blocking=True)
    @DistProfiler.annotate(color="blue", role="old_log_prob_compute")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute log probabilities through TrainingWorker."""
        if self.old_log_prob is None:
            raise RuntimeError("OldLogProbWorker.init_model() must be called before compute_log_prob().")

        output = self.old_log_prob.infer_batch(data)
        return output.cpu() if output is not None else None


@ray.remote
class OldLogProbServer:
    _REQUIRED_REQUEST_KEYS = (
        "input_ids",
        "attention_mask",
        "response_mask",
        "position_ids",
        "prompts",
        "responses",
        "temperature",
    )

    """Server for computing old log probabilities.

    Implements batch control: collects individual requests and batches them
    together before calling infer_batch. When batch_size requests are collected,
    or when a timeout is reached, the batch is dispatched for inference.
    """

    def __init__(
        self,
        old_log_prob_worker_group: RayWorkerGroup,
        old_log_prob_cfg: DictConfig,
    ):
        """Initialize the OldLogProbServer.

        Args:
            old_log_prob_worker_group: The worker group for computing log probabilities.
            batch_size: Number of requests to collect before dispatching a batch.
                Should be a multiple of dp_size * micro_batch_size_per_gpu so
                that full batches are always divisible without padding.
            timeout: Maximum time (in seconds) to wait for a full batch before
                dispatching a partial batch.
            micro_batch_size_per_gpu: The micro-batch size per GPU used during
                inference.  Must match the value set in OldLogProbWorker so that
                prepare_micro_batches never fails with a divisibility assertion.
        """
        self.old_log_prob_worker_group = old_log_prob_worker_group
        self.batch_size = old_log_prob_cfg.get("batch_size", 8)
        self.timeout = old_log_prob_cfg.get("timeout", 10.0)
        self.micro_batch_size_per_gpu = old_log_prob_cfg.get("micro_batch_size_per_gpu", 1)
        self.use_dynamic_bsz = old_log_prob_cfg.get("use_dynamic_bsz", False)

        if "old_log_prob" not in old_log_prob_worker_group._dispatch_info:
            old_log_prob_worker_group._dispatch_info["old_log_prob"] = old_log_prob_worker_group._query_dispatch_info(
                "old_log_prob"
            )
        dp_rank_mapping = old_log_prob_worker_group._dispatch_info["old_log_prob"]
        dp_size = max(dp_rank_mapping) + 1

        if self.use_dynamic_bsz:
            self._min_dispatch_unit = dp_size
        else:
            self._min_dispatch_unit = dp_size * self.micro_batch_size_per_gpu
        self._dp_size = dp_size
        if self.batch_size % self._min_dispatch_unit != 0:
            raise RuntimeError(f"OldLogProbServer {self.batch_size=} is not a multiple of {self._min_dispatch_unit=}. ")

        # Batch control state
        self._pending_requests: list[TensorDict] = []
        self._pending_futures: list[asyncio.Future] = []
        self._batch_lock = asyncio.Lock()
        self._timer_task: asyncio.Task | None = None
        # Ensures only one infer_batch call runs at a time
        self._infer_lock = asyncio.Lock()

    async def compute_old_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute old log probabilities for a single data item.

        This method collects individual requests and batches them together.
        When batch_size requests are collected or timeout is reached,
        the batch is dispatched to infer_batch.

        Args:
            data: A TensorDict with batch_size=1 containing a single data item.

        Returns:
            A TensorDict containing the computed old log probabilities for this item.
        """
        assert data.batch_size[0] == 1, "OldLogProbServer only supports batch size 1"
        missing_keys = [key for key in self._REQUIRED_REQUEST_KEYS if key not in data.keys()]
        if missing_keys:
            raise KeyError(f"OldLogProbServer request missing required keys: {missing_keys}")

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async with self._batch_lock:
            self._pending_requests.append(data)
            self._pending_futures.append(future)

            if len(self._pending_requests) >= self.batch_size:
                # Batch is full, dispatch one full batch immediately.
                await self._dispatch_batch(max_items=self.batch_size)
            elif self._timer_task is None or self._timer_task.done():
                # Start a timeout timer for the first request in a new batch
                self._timer_task = asyncio.create_task(self._timeout_handler())

        # Wait for the result
        return await future

    async def _timeout_handler(self):
        """Wait for the timeout period, then dispatch whatever requests are pending."""
        await asyncio.sleep(self.timeout)
        async with self._batch_lock:
            if self._pending_requests:
                await self._dispatch_batch()

    async def _dispatch_batch(self, max_items: int | None = None):
        """
        Dispatch pending requests as a batch.
        """
        # Cancel the timer if it's still running.
        # If _dispatch_batch is called from the timeout task itself, do not self-cancel;
        if self._timer_task is not None:
            current_task = asyncio.current_task()
            if self._timer_task is current_task:
                self._timer_task = None
            else:
                if not self._timer_task.done():
                    self._timer_task.cancel()
                self._timer_task = None

        # Take either all pending items or only the first max_items.
        if max_items is None:
            requests = self._pending_requests
            futures = self._pending_futures
            self._pending_requests = []
            self._pending_futures = []
        else:
            requests = self._pending_requests[:max_items]
            futures = self._pending_futures[:max_items]
            self._pending_requests = self._pending_requests[max_items:]
            self._pending_futures = self._pending_futures[max_items:]

        # If requests remain queued, keep a timer for the remainder.
        if self._pending_requests and (self._timer_task is None or self._timer_task.done()):
            self._timer_task = asyncio.create_task(self._timeout_handler())

        # Concatenate all single-item TensorDicts into one batch
        batched_data = TensorDict.cat(requests, dim=0)

        try:
            # Acquire infer_lock to ensure only one infer_batch runs at a time.
            loop = asyncio.get_event_loop()
            async with self._infer_lock:
                batched_output = await loop.run_in_executor(None, self.infer_batch, batched_data)

            # Split the output and resolve each future
            for i, future in enumerate(futures):
                if not future.done():
                    individual_result = batched_output[i : i + 1]
                    future.set_result(individual_result)
        except Exception as e:
            import traceback

            traceback.print_exc()
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def drain_and_load_weights(self):
        """
        Wait for all pending requests to finish, then load staged weights into the engine.
        """
        async with self._batch_lock:
            if self._pending_requests:
                await self._dispatch_batch()

            # Wait for any in-flight inference to finish, then load new weights
            async with self._infer_lock:
                self.old_log_prob_worker_group.load_staged_weights()

    def infer_batch(self, data: TensorDict) -> TensorDict:
        """
        Call the worker group to compute log probabilities for a batch.
        """
        n_real = len(data)

        # Pad batch up to the nearest multiple of _min_dispatch_unit.
        remainder = n_real % self._min_dispatch_unit
        if remainder != 0:
            n_pad = self._min_dispatch_unit - remainder
            dummy = TensorDict.cat([data[0:1]] * n_pad, dim=0)
            data = TensorDict.cat([data, dummy], dim=0)

        global_token_num = torch.sum(data["attention_mask"], dim=-1).tolist()
        tu.assign_non_tensor(data, global_token_num=global_token_num)
        data = left_right_2_no_padding(data)
        tu.assign_non_tensor(
            data,
            calculate_entropy=True,
            compute_loss=False,
        )

        output = self.old_log_prob_worker_group.compute_log_prob(data)
        if isinstance(output, DataProtoFuture):
            output = output.get()
        if output is None:
            raise RuntimeError("OldLogProbWorkerGroup.compute_log_prob returned None.")

        if "log_probs" not in output.keys():
            raise KeyError(f"Expected 'log_probs' in old_log_prob output, got keys: {list(output.keys())}")
        log_probs = no_padding_2_padding(tu.get(output, "log_probs"), data).float()
        entropy = tu.get(output, "entropy", default=None)
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, data).float()

        # Discard dummy padding rows; keep only the real results.
        log_probs = log_probs[:n_real]

        recovered_td = tu.get_tensordict({"log_probs": log_probs.float()})
        return recovered_td
