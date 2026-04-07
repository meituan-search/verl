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
        self._gpu_lock: asyncio.Lock | None = None

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
            self.config.old_log_prob.pop("batch_size", None)
            self.config.old_log_prob.pop("timeout", None)

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

    def _get_gpu_lock(self) -> asyncio.Lock:
        """Return the per-actor GPU lock, creating it lazily inside the event loop."""
        if self._gpu_lock is None:
            self._gpu_lock = asyncio.Lock()
        return self._gpu_lock

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """
        Receive weights from trainer via checkpoint engine and stage to CPU.
        """
        if self.checkpoint_engine is None:
            return
        print(f"[OldLogProbWorker] update_weights")
        # Receive weights and stage to CPU
        async with self._get_gpu_lock():
            staged = {}
            async for name, tensor in self.checkpoint_engine.receive_weights():
                staged[name] = tensor.clone().to("cpu", non_blocking=False)
                logger.info(f"[OldLogProbWorker] update_weights, {name=}, shape={tensor.shape}")

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
    async def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute log probabilities through TrainingWorker."""
        if self.old_log_prob is None:
            raise RuntimeError("OldLogProbWorker.init_model() must be called before compute_log_prob().")

        async with self._get_gpu_lock():
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
            old_log_prob_cfg: Configuration containing:
                - batch_size: Number of requests to collect before dispatching.
                  Must be a multiple of dp_size * micro_batch_size_per_gpu.
                - timeout: Maximum time (seconds) to wait for a full batch.
                - micro_batch_size_per_gpu: Micro-batch size per GPU for inference.
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

        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None
        # Inference lock to serialize worker_group calls
        self._infer_lock = asyncio.Lock()
        self._shutdown = False
        self._drain_done = asyncio.Event()
        self._drain_done.set()  # Initially open (not draining)

        self._start_consumer()

    def _start_consumer(self):
        """Start the background batch consumer task."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._batch_consumer())
            logger.info("OldLogProbServer: Started batch consumer task")

    async def _batch_consumer(self):
        """
        Background task that continuously collects and dispatches batches.

        1. Wait until the drain gate is open (not in a drain/weight-load window)
        2. Try to collect batch_size requests within the timeout window;
        3. If timeout expires dispatch whatever is available
        """
        while not self._shutdown:
            batch_requests = []
            batch_futures = []

            # 1. Wait until drain gate is open before starting a new collection round.
            await self._drain_done.wait()
            try:
                # 2: Collect up to batch_size requests with timeout window
                deadline = asyncio.get_event_loop().time() + self.timeout
                drain_triggered = False

                while len(batch_requests) < self.batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break

                    try:
                        # Wait for next request with remaining timeout
                        item = await asyncio.wait_for(self._request_queue.get(), timeout=remaining_time)

                        # After every await, check whether a drain has started.
                        if not self._drain_done.is_set():
                            # Put back items
                            for req, fut in zip(batch_requests, batch_futures, strict=False):
                                await self._request_queue.put((req, fut))
                            await self._request_queue.put(item)
                            batch_requests.clear()
                            batch_futures.clear()
                            drain_triggered = True
                            break

                        batch_requests.append(item[0])
                        batch_futures.append(item[1])

                        # Batch is full — dispatch immediately without waiting for timeout
                        if len(batch_requests) >= self.batch_size:
                            break

                    except asyncio.TimeoutError:
                        # Timeout window expired; dispatch whatever we have
                        break

                if drain_triggered:
                    # Wait for the drain to finish before retrying.
                    await self._drain_done.wait()
                    continue

                # 3. Dispatch the collected batch.
                if batch_requests:
                    await self._execute_batch(batch_requests, batch_futures)

            except Exception as e:
                logger.exception(f"OldLogProbServer: Error in batch consumer: {e}")
                raise

    async def _execute_batch(self, requests: list[TensorDict], futures: list[asyncio.Future]):
        """
        Execute inference for a batch of requests and resolve futures.
        """
        n_real = len(requests)
        if n_real != self.batch_size:
            logger.debug(f"OldLogProbServer: Dispatching partial batch {n_real=} (target={self.batch_size})")

        batched_data = TensorDict.cat(requests, dim=0)

        try:
            loop = asyncio.get_event_loop()
            # Use _infer_lock to serialize inference calls
            async with self._infer_lock:
                batched_output = await loop.run_in_executor(None, self.infer_batch, batched_data)

            # Resolve all futures with their corresponding results
            for i, future in enumerate(futures):
                if not future.done():
                    individual_result = batched_output[i : i + 1]
                    future.set_result(individual_result)

        except Exception as e:
            logger.exception(f"OldLogProbServer: Batch inference failed: {e}")
            # Propagate error to all futures in this batch
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def compute_old_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute old log probabilities for a single data item."""
        if self._shutdown:
            raise RuntimeError("OldLogProbServer is shutting down, cannot accept new requests")
        assert data.batch_size[0] == 1, "OldLogProbServer only supports batch size 1"
        missing_keys = [key for key in self._REQUIRED_REQUEST_KEYS if key not in data.keys()]
        if missing_keys:
            raise KeyError(f"OldLogProbServer request missing required keys: {missing_keys}")

        # Wait until any in-progress drain/weight-load has finished before enqueuing.
        await self._drain_done.wait()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._request_queue.put((data, future))

        return await future

    async def drain_and_load_weights(self):
        """Flush requests already in the queue with the current weights, load new weights."""
        logger.info("OldLogProbServer: Starting drain for weight update")

        # _drain_done.wait() will block
        self._drain_done.clear()

        async with self._infer_lock:
            # flush requests already in the queue with the old weights.
            pre_drain_requests = []
            pre_drain_futures = []
            while not self._request_queue.empty():
                try:
                    item = self._request_queue.get_nowait()
                    pre_drain_requests.append(item[0])
                    pre_drain_futures.append(item[1])
                except asyncio.QueueEmpty:
                    break

            if pre_drain_requests:
                logger.info(
                    f"OldLogProbServer: Flushing {len(pre_drain_requests)} pre-drain requests with current weights"
                )
                loop = asyncio.get_event_loop()
                batched_data = TensorDict.cat(pre_drain_requests, dim=0)
                try:
                    batched_output = await loop.run_in_executor(None, self.infer_batch, batched_data)
                    for i, future in enumerate(pre_drain_futures):
                        if not future.done():
                            future.set_result(batched_output[i : i + 1])
                except Exception as e:
                    logger.exception("OldLogProbServer: pre-drain flush failed")
                    for future in pre_drain_futures:
                        if not future.done():
                            future.set_exception(e)

            # Step 4: load new weights with the lock still held.
            self.old_log_prob_worker_group.load_staged_weights()
            logger.info("OldLogProbServer: Weights loaded successfully")
        self._drain_done.set()
        logger.info("OldLogProbServer: Drain complete, consumer resuming normal operation")

    async def shutdown(self):
        """Shutdown the server."""
        logger.info("OldLogProbServer: Shutting down")
        self._shutdown = True

        # Cancel consumer task
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        # Fail any remaining requests in queue
        while not self._request_queue.empty():
            try:
                _, future = self._request_queue.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError("Server shutdown"))
            except asyncio.QueueEmpty:
                break

        logger.info("OldLogProbServer: Shutdown complete")

    def infer_batch(self, data: TensorDict) -> TensorDict:
        """Call the worker group to compute log probabilities for a batch."""
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
