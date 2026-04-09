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
from typing import Generator

import ray
import torch
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.protocol import DataProtoFuture
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_torch_device, is_torch_npu_available
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.utils.profiler.performance import log_gpu_memory_usage
from verl.workers.config import HFModelConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker, _with_routing_replay_flag
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class OldLogProbServerAdapter(BaseRollout):
    """BaseRollout adapter that wraps a TrainingWorker for old log probability computation."""

    def __init__(
        self,
        full_config: DictConfig,
        **kwargs,
    ):
        self.config = None
        self.model_config = None
        self.device_mesh = None

        self._full_config = full_config
        self._training_worker: TrainingWorker = None

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # replica_rank and is_leader_rank are consumed by CheckpointEngineWorker
        self.replica_rank = rank // world_size
        self.is_leader_rank = rank % world_size == 0

        self.enable_routing_replay = (
            self._full_config.old_log_prob.strategy == "megatron"
            and self._full_config.old_log_prob.router_replay.mode != "disabled"
        )

    def init_model(self):
        """Initialize the internal TrainingWorker and register dispatch info."""
        assert self._full_config is not None, "full_config must be provided to init_model()"

        model_config: HFModelConfig = omega_conf_to_dataclass(self._full_config.actor_rollout_ref.model)

        # Remap old_log_prob-specific config keys to ppo-compatible names expected
        # by TrainingWorkerConfig / EngineConfig.
        with open_dict(self._full_config.old_log_prob):
            self._full_config.old_log_prob.pop("enable_standalone", None)
            self._full_config.old_log_prob.pop("nnodes", None)
            self._full_config.old_log_prob.pop("n_gpus_per_node", None)
            self._full_config.old_log_prob.pop("batch_size", None)
            self._full_config.old_log_prob.pop("timeout", None)

            self._full_config.old_log_prob.ppo_mini_batch_size = self._full_config.old_log_prob.micro_batch_size_per_gpu
            self._full_config.old_log_prob.ppo_micro_batch_size_per_gpu = self._full_config.old_log_prob.pop(
                "micro_batch_size_per_gpu", None
            )
            self._full_config.old_log_prob.ppo_max_token_len_per_gpu = self._full_config.old_log_prob.pop(
                "max_token_len_per_gpu", None
            )

        old_log_prob_config = omega_conf_to_dataclass(self._full_config.old_log_prob)
        old_log_prob_config.model_config = model_config

        training_worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=old_log_prob_config.model_config,
            engine_config=old_log_prob_config.engine,
        )

        training_worker_config.engine_config.use_dynamic_bsz = self._full_config.old_log_prob.use_dynamic_bsz
        training_worker_config.engine_config.infer_max_token_len_per_gpu = (
            self._full_config.old_log_prob.ppo_max_token_len_per_gpu
        )
        training_worker_config.engine_config.infer_micro_batch_size_per_gpu = (
            self._full_config.old_log_prob.ppo_micro_batch_size_per_gpu
        )
        training_worker_config.engine_config.max_token_len_per_gpu = (
            self._full_config.old_log_prob.ppo_max_token_len_per_gpu
        )
        training_worker_config.engine_config.micro_batch_size_per_gpu = (
            self._full_config.old_log_prob.ppo_micro_batch_size_per_gpu
        )
        training_worker_config.engine_config.use_remove_padding = model_config.use_remove_padding
        if self._full_config.old_log_prob.use_dynamic_bsz:
            assert self._full_config.old_log_prob.ppo_max_token_len_per_gpu is not None
        else:
            assert self._full_config.old_log_prob.ppo_micro_batch_size_per_gpu is not None

        self._training_worker = TrainingWorker(config=training_worker_config)
        self._training_worker.reset()
        log_gpu_memory_usage("[OldLogProbServerAdapter] After init model", logger=logger)
        aggressive_empty_cache(force_sync=True)

    # ------------------------------------------------------------------
    # BaseRollout interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        **kwargs,
    ):
        """Receive weights from the checkpoint engine"""
        if self._training_worker is None:
            return
        await self._training_worker.engine.set_param_from_async_generator(weights)
        get_torch_device().empty_cache()
        logger.info("[OldLogProbServerAdapter] loaded weights into engine")

    async def resume(self, tags: list[str]):
        """No-op for OldLogProbServerAdapter."""
        pass

    async def release(self):
        """Release GPU memory (no-op for TrainingWorker; override if needed)."""
        get_torch_device().empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Run a forward pass through the TrainingWorker engine."""
        if self._training_worker is None:
            raise RuntimeError("OldLogProbServerAdapter.init_model() must be called first.")
        output = self._training_worker.infer_batch(data)
        return output.cpu() if output is not None else None

    def get_dispatch_collect(self):
        """Expose the dispatch/collect info from the underlying TrainingWorker."""
        return self._training_worker.get_dispatch_collect()


class OldLogProbWorker(CheckpointEngineWorker, DistProfilerExtension):
    """Worker for computing old log probabilities."""

    def __init__(self, config: DictConfig, role: str = "old_log_prob"):
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        from verl.utils.distributed import initialize_global_process_group_ray

        initialize_global_process_group_ray(timeout_second=None)  # default: gloo+nccl

        server_adapter = OldLogProbServerAdapter(full_config=config)

        CheckpointEngineWorker.__init__(
            self,
            rollout_config=rollout_config,
            model_config=model_config,
            server_adapter=server_adapter,
        )

        self.config = config
        self.role = role

        # Profiler setup
        omega_profiler_config = config.old_log_prob.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

    @property
    def _server_adapter(self) -> OldLogProbServerAdapter:
        return self.server_adapter  # type: ignore[return-value]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialise the TrainingWorker inside the server adapter and register dispatch info."""
        self._server_adapter.init_model()
        self.set_dispatch_collect(mesh_name="old_log_prob", **self._server_adapter.get_dispatch_collect())

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="old_log_prob"))
    @DistProfiler.annotate(color="blue", role="old_log_prob_compute")
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute log probabilities through the TrainingWorker engine."""
        return self._server_adapter.compute_log_prob(data)


@ray.remote
class OldLogProbServer:
    """Server for computing old log probabilities.

    Responsibilities:
    - Collect individual inference requests into batches.
    - Dispatch batches to OldLogProbWorker via the worker group.
    - Gate new requests during weight updates (drain → NCCL → load → resume).
    """

    _REQUIRED_REQUEST_KEYS = (
        "input_ids",
        "attention_mask",
        "response_mask",
        "position_ids",
        "prompts",
        "responses",
        "temperature",
    )

    def __init__(
        self,
        old_log_prob_worker_group: RayWorkerGroup,
        old_log_prob_cfg: DictConfig,
    ):
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
            raise RuntimeError(f"OldLogProbServer {self.batch_size=} is not a multiple of {self._min_dispatch_unit=}.")

        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None
        self._infer_lock = asyncio.Lock()
        self._shutdown = False
        self._serving: asyncio.Event = asyncio.Event()
        self._serving.set()  # Initially serving (not draining)

        self._start_consumer()

    # ------------------------------------------------------------------
    # Internal batch consumer
    # ------------------------------------------------------------------

    def _start_consumer(self):
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._batch_consumer())
            logger.info("OldLogProbServer: started batch consumer task")

    async def _batch_consumer(self):
        """Continuously collect requests and dispatch them as batches.

        Algorithm:
        1. Wait until the _serving gate is open.
        2. Collect up to ``batch_size`` requests within the timeout window.
        3. On timeout or full batch, dispatch immediately.
        4. If a drain starts mid-collection, put items back and wait.
        """
        while not self._shutdown:
            batch_requests = []
            batch_futures = []

            await self._serving.wait()
            try:
                deadline = asyncio.get_event_loop().time() + self.timeout
                drain_triggered = False

                while len(batch_requests) < self.batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(self._request_queue.get(), timeout=remaining_time)

                        if not self._serving.is_set():
                            # Drain started mid-collection — put everything back.
                            for req, fut in zip(batch_requests, batch_futures, strict=False):
                                await self._request_queue.put((req, fut))
                            await self._request_queue.put(item)
                            batch_requests.clear()
                            batch_futures.clear()
                            drain_triggered = True
                            break

                        batch_requests.append(item[0])
                        batch_futures.append(item[1])

                        if len(batch_requests) >= self.batch_size:
                            break

                    except asyncio.TimeoutError:
                        break

                if drain_triggered:
                    await self._serving.wait()
                    continue

                if batch_requests:
                    await self._execute_batch(batch_requests, batch_futures)

            except Exception as e:
                logger.exception(f"OldLogProbServer: error in batch consumer: {e}")
                raise

    async def _execute_batch(self, requests: list[TensorDict], futures: list[asyncio.Future]):
        n_real = len(requests)
        if n_real != self.batch_size:
            logger.debug(f"OldLogProbServer: dispatching partial batch {n_real=} (target={self.batch_size})")

        batched_data = TensorDict.cat(requests, dim=0)
        try:
            loop = asyncio.get_event_loop()
            async with self._infer_lock:
                batched_output = await loop.run_in_executor(None, self.infer_batch, batched_data)

            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result(batched_output[i : i + 1])

        except Exception as e:
            logger.exception(f"OldLogProbServer: batch inference failed: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def infer_batch(self, data: TensorDict) -> TensorDict:
        """Pad the batch to a multiple of ``_min_dispatch_unit`` and call the worker group."""
        n_real = len(data)
        remainder = n_real % self._min_dispatch_unit
        if remainder != 0:
            n_pad = self._min_dispatch_unit - remainder
            dummy = TensorDict.cat([data[0:1]] * n_pad, dim=0)
            data = TensorDict.cat([data, dummy], dim=0)

        global_token_num = torch.sum(data["attention_mask"], dim=-1).tolist()
        tu.assign_non_tensor(data, global_token_num=global_token_num)
        data = left_right_2_no_padding(data)
        tu.assign_non_tensor(data, calculate_entropy=False, compute_loss=False)

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

        # Discard dummy padding rows.
        log_probs = log_probs[:n_real]

        return tu.get_tensordict({"log_probs": log_probs.float()})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compute_old_log_prob(self, data: TensorDict) -> TensorDict:
        """Enqueue a single request and wait for its result."""
        if self._shutdown:
            raise RuntimeError("OldLogProbServer is shutting down, cannot accept new requests")
        assert data.batch_size[0] == 1, "OldLogProbServer only supports batch size 1"
        missing_keys = [key for key in self._REQUIRED_REQUEST_KEYS if key not in data.keys()]
        if missing_keys:
            raise KeyError(f"OldLogProbServer request missing required keys: {missing_keys}")

        await self._serving.wait()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._request_queue.put((data, future))
        return await future

    # ------------------------------------------------------------------
    # Weight-update protocol: called by OldLogProbReplica.sleep/wake_up
    # ------------------------------------------------------------------

    async def pause_serving(self):
        """
        Stop accepting new requests, wait for any in-flight inference to finish,
        then flush queued requests with current weights.
        """
        logger.info("OldLogProbServer: pause_serving — starting drain")

        # Block new requests from entering the queue and prevent _batch_consumer
        # from starting a new _execute_batch after the current one finishes.
        self._serving.clear()

        # Wait for any in-flight _execute_batch to complete.
        async with self._infer_lock:
            pass

        # Flush requests already sitting in the queue using current (old) weights.
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
            logger.info(f"OldLogProbServer: flushing {len(pre_drain_requests)} pre-drain requests with current weights")
            await self._execute_batch(pre_drain_requests, pre_drain_futures)

        logger.info("OldLogProbServer: pause_serving done")

    async def resume_serving(self):
        """Re-open the request gate after new weights have been loaded."""
        self._serving.set()
        logger.info("OldLogProbServer: resume_serving done, consumer resuming")

    async def shutdown(self):
        """Shut down the server."""
        logger.info("OldLogProbServer: shutting down")
        self._shutdown = True

        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        while not self._request_queue.empty():
            try:
                _, future = self._request_queue.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError("Server shutdown"))
            except asyncio.QueueEmpty:
                break

        logger.info("OldLogProbServer: shutdown complete")


class OldLogProbReplica(RolloutReplica):
    """RolloutReplica implementation for the old log probability inference server."""

    def __init__(
        self,
        replica_rank: int,
        full_config: DictConfig,
        worker_cls,
    ):
        # old_log_prob uses a plain DP training engine (no TP/PP), so world_size is
        # simply n_gpus_per_node * nnodes.  We bypass RolloutReplica.__init__'s
        # world_size formula (which reads rollout TP/PP) and set the fields directly.
        self.replica_rank = replica_rank
        self.config = None  # not used by OldLogProbReplica
        self.model_config = None  # not used by OldLogProbReplica

        old_log_prob_cfg = full_config.old_log_prob
        n_gpus_per_node = int(old_log_prob_cfg.n_gpus_per_node)
        nnodes = int(old_log_prob_cfg.nnodes)
        self.world_size = n_gpus_per_node * nnodes
        self.gpus_per_node = n_gpus_per_node
        self.gpus_per_replica_node = n_gpus_per_node
        self.nnodes = nnodes
        self.is_reward_model = False

        self.rollout_mode = None
        self.workers = []
        self.resource_pool = None
        self.bundle_indices = []
        self.servers = []
        self._server_address = None
        self._server_handle = None

        self._full_config = full_config
        self._worker_cls = worker_cls
        self._worker_group: RayWorkerGroup = None

    # ------------------------------------------------------------------
    # RolloutReplica interface
    # ------------------------------------------------------------------

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Return the Ray actor class for OldLogProbWorker."""
        return RayClassWithInitArgs(
            cls=ray.remote(self._worker_cls),
            config=self._full_config,
            role="old_log_prob",
        )

    async def init_standalone(self):
        """Self-allocate a resource pool, spawn workers, init model, and create server."""
        self.rollout_mode = RolloutMode.STANDALONE

        # 1. Create an independent Ray resource pool for this replica.
        resource_pool_name = f"old_log_prob_pool_{self.replica_rank}"
        resource_pool_spec = {resource_pool_name: [self.gpus_per_replica_node] * self.nnodes}
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

        # 2. OldLogProbWorker RayWorkerGroup.
        self._worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"old_log_prob_standalone_{self.replica_rank}",
            device_name="cuda" if not is_torch_npu_available(check_device=False) else "npu",
        )
        self.workers = self._worker_group.workers

        # 3. init_model + create OldLogProbServer.
        await self.launch_servers()

    async def launch_servers(self):
        """Call init_model on every worker, then create the OldLogProbServer actor."""
        self._worker_group.init_model()

        server = OldLogProbServer.options(
            name=f"old_log_prob_server_{self.replica_rank}",
        ).remote(
            old_log_prob_worker_group=self._worker_group,
            old_log_prob_cfg=self._full_config.old_log_prob,
        )
        self.servers = [server]

    async def abort_all_requests(self):
        """No-op for OldLogProbReplica."""
        pass

    async def resume_generation(self):
        """No-op for OldLogProbReplica."""
        pass

    async def sleep(self):
        """Pause serving: stop accepting new requests and flush queued ones with current weights."""
        if self.servers:
            await self.servers[0].pause_serving.remote()

    async def wake_up(self):
        """Re-open the request gate after new weights have been loaded."""
        if self.servers:
            await self.servers[0].resume_serving.remote()

    async def shutdown(self):
        """Gracefully shut down the OldLogProbServer."""
        if self.servers:
            await self.servers[0].shutdown.remote()
            self.servers = []
