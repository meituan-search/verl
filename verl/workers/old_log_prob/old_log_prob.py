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
from verl.workers.config import HFModelConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker, _with_routing_replay_flag
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)


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

            self.config.old_log_prob.ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            self.config.old_log_prob.ppo_micro_batch_size = self.config.old_log_prob.pop(
                "log_prob_micro_batch_size", None
            )
            self.config.old_log_prob.ppo_micro_batch_size_per_gpu = self.config.old_log_prob.pop(
                "log_prob_micro_batch_size_per_gpu", None
            )
            self.config.old_log_prob.use_dynamic_bsz = self.config.old_log_prob.pop("log_prob_use_dynamic_bsz", False)
            self.config.old_log_prob.ppo_max_token_len_per_gpu = self.config.old_log_prob.pop(
                "log_prob_max_token_len_per_gpu", None
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
            self.config.old_log_prob.ppo_max_token_len_per_gpu
        )
        training_worker_config.engine_config.infer_micro_batch_size_per_gpu = (
            self.config.old_log_prob.ppo_micro_batch_size_per_gpu
        )
        training_worker_config.engine_config.use_remove_padding = model_config.use_remove_padding

        self.old_log_prob = TrainingWorker(config=training_worker_config)
        self.old_log_prob.reset()
        self.set_dispatch_collect(mesh_name="old_log_prob", **self.old_log_prob.get_dispatch_collect())

        # Build checkpoint engine (as receiver, is_master=False)
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            backend, is_master=False, bucket_size=bucket_size, **engine_kwargs
        )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """Receive weights from trainer via checkpoint engine and stage to CPU.

        The weights are NOT loaded into the model engine immediately. Instead they are
        stored in CPU memory as a staging area. Call load_staged_weights() later
        (after OldLogProbServer finishes all pending requests) to actually load them
        into the engine.
        """
        print("[old_log_prob] update_weights")
        if self.checkpoint_engine is None:
            return
        print("[old_log_prob] update_weights1")
        # Receive weights and stage to CPU
        staged = {}
        async for name, tensor in self.checkpoint_engine.receive_weights():
            # Must clone: tensor is a view into the double-buffer which will be overwritten
            staged[name] = tensor.clone().to("cpu", non_blocking=True)
            print(f"[old_log_prob] update_weights, {name=}, shape={tensor.shape}")

        torch.cuda.synchronize()
        self._staged_state_dict = staged
        print(f"Rank {self.rank}: staged {len(staged)} params to CPU")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_staged_weights(self):
        """Load staged CPU weights into the model engine.

        Should be called after OldLogProbServer has finished processing all pending
        requests, so that inference is not interrupted by the weight update.
        """
        print("[old_log_prob] load_staged_weights")
        if self._staged_state_dict is None or self.old_log_prob is None:
            return
        print("[old_log_prob] load_staged_weights 1")

        # Load into sharded model (handles FSDP DTensor reshard / Megatron TP/PP reshard)
        self.old_log_prob.engine.set_param(self._staged_state_dict)

        # Free staging area
        self._staged_state_dict = None
        torch.cuda.empty_cache()
        print(f"Rank {self.rank}: loaded staged weights into engine")

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Proxy method for CheckpointEngineManager to call prepare/init_process_group/finalize."""
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="old_log_prob"), blocking=True)
    @DistProfiler.annotate(color="blue", role="old_log_prob_compute")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute log probabilities through the internal TrainingWorker."""
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

    def __init__(self, old_log_prob_worker_group: RayWorkerGroup, batch_size: int = 8, timeout: float = 1.0):
        """Initialize the OldLogProbServer.

        Args:
            old_log_prob_worker_group: The worker group for computing log probabilities.
            batch_size: Number of requests to collect before dispatching a batch.
            timeout: Maximum time (in seconds) to wait for a full batch before
                dispatching a partial batch.
        """
        self.old_log_prob_worker_group = old_log_prob_worker_group
        self.batch_size = batch_size
        self.timeout = timeout

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
        self._validate_request(data)

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async with self._batch_lock:
            self._pending_requests.append(data)
            self._pending_futures.append(future)

            if len(self._pending_requests) >= self.batch_size:
                # Batch is full, dispatch immediately
                await self._dispatch_batch()
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

    async def _dispatch_batch(self):
        """Dispatch the current pending requests as a batch.

        Must be called while holding self._batch_lock.
        Concatenates all pending TensorDicts, calls infer_batch,
        and resolves each caller's future with their individual result.
        """
        # Cancel the timer if it's still running.
        # If _dispatch_batch is called from the timeout task itself, do not
        # self-cancel; that would abort this coroutine before futures are resolved.
        if self._timer_task is not None:
            current_task = asyncio.current_task()
            if self._timer_task is current_task:
                self._timer_task = None
            else:
                if not self._timer_task.done():
                    self._timer_task.cancel()
                self._timer_task = None

        # Take the current pending requests and futures
        requests = self._pending_requests
        futures = self._pending_futures
        self._pending_requests = []
        self._pending_futures = []

        # Concatenate all single-item TensorDicts into one batch
        batched_data = TensorDict.cat(requests, dim=0)

        try:
            # Acquire infer_lock to ensure only one infer_batch runs at a time.
            # Run infer_batch in a thread executor to avoid blocking the asyncio event loop
            # (infer_batch issues a blocking Ray RPC on the worker group).
            loop = asyncio.get_event_loop()
            async with self._infer_lock:
                batched_output = await loop.run_in_executor(None, self.infer_batch, batched_data)

            # Split the output and resolve each future
            for i, future in enumerate(futures):
                if not future.done():
                    individual_result = batched_output[i : i + 1]
                    future.set_result(individual_result)
        except Exception as e:
            # If inference fails, propagate the exception to all waiters
            logger.error(f"OldLogProbServer infer_batch failed: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    # TODO: called once all need response finished
    # To ensure consistency, oldlogprobserver should wait for all pending requests to finish
    # drain_and_load_weights should be called after all pending requests are finished
    async def drain_and_load_weights(self):
        """Wait for all pending requests to finish, then load staged weights into the engine.

        This ensures no inference is running when the model weights are being updated.
        The flow is:
        1. Stop accepting new batches (acquire infer_lock).
        2. Dispatch any remaining pending requests.
        3. Call load_staged_weights on the worker group.
        4. Release the lock so new requests can proceed with updated weights.
        """
        # Acquire both locks to ensure no new requests are batched and no inference is running
        async with self._batch_lock:
            # Dispatch any remaining pending requests with old weights
            if self._pending_requests:
                await self._dispatch_batch()

        # Wait for any in-flight inference to finish, then load new weights
        async with self._infer_lock:
            self.old_log_prob_worker_group.load_staged_weights()

    def _validate_request(self, data: TensorDict) -> None:
        missing_keys = [key for key in self._REQUIRED_REQUEST_KEYS if key not in data.keys()]
        if missing_keys:
            raise KeyError(f"OldLogProbServer request missing required keys: {missing_keys}")

    def _prepare_for_log_prob_compute(self, data: TensorDict) -> TensorDict:
        # Keep preprocessing consistent with ray_trainer._compute_old_log_prob.
        data = left_right_2_no_padding(data)

        calculate_entropy = bool(tu.get(data, "calculate_entropy", default=False))
        tu.assign_non_tensor(data, calculate_entropy=calculate_entropy, compute_loss=False)
        return data

    def _recover_from_no_padding(self, output: TensorDict, data: TensorDict) -> TensorDict:
        if output is None:
            raise RuntimeError("OldLogProbWorkerGroup.compute_log_prob returned None.")

        if "log_probs" not in output.keys():
            raise KeyError(f"Expected 'log_probs' in old_log_prob output, got keys: {list(output.keys())}")

        recovered = {
            "log_probs": no_padding_2_padding(tu.get(output, "log_probs"), data).float(),
        }

        entropy = tu.get(output, "entropy", default=None)
        if entropy is not None:
            recovered["entropy"] = no_padding_2_padding(entropy, data).float()

        recovered_td = TensorDict(recovered, batch_size=data.batch_size)
        metrics = tu.get(output, "metrics", default=None)
        if metrics is not None:
            tu.assign_non_tensor(recovered_td, metrics=metrics)

        return recovered_td

    def infer_batch(self, data: TensorDict) -> TensorDict:
        """Call the worker group to compute log probabilities for a batch.

        Args:
            data: A batched TensorDict containing multiple data items.

        Returns:
            A TensorDict containing the computed old log probabilities.
        """
        data = self._prepare_for_log_prob_compute(data)

        output = self.old_log_prob_worker_group.compute_log_prob(data)
        # compute_log_prob is dispatched with blocking=False and may return
        # a DataProtoFuture wrapper. Resolve it before slicing per request.
        if isinstance(output, DataProtoFuture):
            output = output.get()

        return self._recover_from_no_padding(output, data)
