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
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, TypedDict

import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, RolloutReplica, get_rollout_class


class TensorMeta(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class CheckpointEngineRegistry:
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        """Get the checkpoint engine class.

        Args:
            backend: The backend of the checkpoint engine.

        Returns:
            The checkpoint engine class.
        """
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend} not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from trainer to rollout.

    In trainer process:
    >>> trainer = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(trainer.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare checkpoint engine before each step send_weights/receive_weights.

        1. Allocate weight bucket.
        2. [Optional] Register weight bucket for RDMA.
        3. Return metadata to build communication topology: master ip:port, register RDMA description, etc.

        Args:
            worker_group: The worker group that the checkpoint engine will be used.

        Returns:
            A dictionary that contains the metadata of the worker group.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
        extra_groups_world_size_list: Optional[list[int]] = None,
    ) -> (
        tuple[dict[str, list[Any]], dict[str, list[Any]]]
        | tuple[dict[str, list[Any]], dict[str, list[Any]], list[dict[str, list[Any]]]]
    ):
        """Build communication topology between all workers.

        Args:
            trainer_world_size: The world size of the trainer worker group.
            rollout_world_size: The world size of the rollout replica.
            metadata: A list of metadata `prepare` from all workers.
            extra_groups_world_size_list: The world size of the extra groups, optional.

        Returns:
            A tuple of two or three elements:
            - A dictionary that contains the communication topology for trainer worker group.
            - A dictionary that contains the communication topology for rollout worker group.
            - Optional: a list of dictionaries that contains the communication topology for extra groups.
            Each dict value should be a list argument equal to the world size of the worker group to dispatch to
            `init_process_group`.

            ```
            world_size = rollout.world_size + trainer.world_size + sum(extra_groups_world_size_list)
            kwargs = {
                "rank": list(range(world_size)),
                "world_size": [world_size] * world_size,
                "master_metadata": [metadata[0]] * world_size,
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Init process group for checkpoint engine.

        Args:
            **kwargs: Keyword arguments from `build_topology`.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights.

        1. Free weight bucket.
        1. [Optional] Deregister weight bucket for RDMA.
        2. [Optional] Destroy process group.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for trainer and rollout colocated on same GPU.

    In trainer process:
    >>> engine = ColocatedCheckpointEngine()
    >>> trainer = Trainer()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(trainer.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def __init__(self, bucket_size: int, is_master: bool = False) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        raise NotImplementedError

    def init_process_group(self, **kwargs):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None


class CheckpointEngineWorker(Worker):
    """CheckpointEngineWorker colocated with inference engine's WorkerProc on same GPU.

    Args:
        rollout_config: The rollout configuration.
        model_config: The model configuration.
        server_adapter: The server adapter to update weights.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rollout_config = rollout_config
        self.model_config = model_config

        self.server_adapter: BaseRollout = server_adapter
        backend = self.rollout_config.checkpoint_engine.backend
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        self.checkpoint_engine: CheckpointEngine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )
        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        weights = self.checkpoint_engine.receive_weights()
        await self.server_adapter.update_weights(weights, global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_replica_rank(self) -> int:
        """Get replica rank from the underlying rollout server adapter."""
        return self.server_adapter.replica_rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def is_leader_rank(self) -> bool:
        """Get leader rank flag from the underlying rollout server adapter."""
        return self.server_adapter.is_leader_rank


_worker_cls = ray.remote(CheckpointEngineWorker)


class CheckpointEngineManager:
    """Checkpoint engine manager to coordinate weight synchronization between trainer and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In trainer, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.

    ```
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
    │ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
    │ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
    │    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
    | ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
    │ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
    │ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
    └────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
         v                                        |    |    |    |    |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
    ```

    Args:
        config: The checkpoint engine config.
        trainer: The trainer worker group.
        replicas: The list of rollout replicas.
        extra_groups: Optional extra worker groups that should join weight sync.
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        trainer: RayWorkerGroup,
        replicas: list[RolloutReplica],
        extra_groups: Optional[list[RayWorkerGroup]] = None,
    ) -> None:
        self.config = config
        self.backend = config.backend
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        self.trainer = trainer
        self.replicas = replicas
        self.extra_groups = list(extra_groups) if extra_groups else []
        # Keep references to background tasks
        self._background_tasks: set[asyncio.Task] = set()

    def build_process_group(self, rollout: RayWorkerGroup):
        """Build process group for trainer and rollout replicas."""
        trainer = self.trainer

        # 1. prepare all workers
        prepare_refs = trainer.execute_checkpoint_engine(
            ["prepare"] * trainer.world_size
        ) + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        for wg in self.extra_groups:
            prepare_refs += wg.execute_checkpoint_engine(["prepare"] * wg.world_size)
        metadata = ray.get(prepare_refs)

        # 2. build communication topology between all workers
        if self.extra_groups:
            try:
                topology = self.backend_cls.build_topology(
                    trainer.world_size,
                    rollout.world_size,
                    metadata,
                    extra_groups_world_size_list=[wg.world_size for wg in self.extra_groups],
                )
            except TypeError as e:
                if "extra_groups_world_size_list" not in str(e):
                    raise
                raise TypeError(
                    f"Checkpoint engine backend `{self.backend}` does not support extra_groups topology."
                ) from e
        else:
            topology = self.backend_cls.build_topology(trainer.world_size, rollout.world_size, metadata)

        if len(topology) == 2:
            trainer_kwargs, rollout_kwargs = topology
            extra_groups_kwargs_list = []
        elif len(topology) == 3:
            trainer_kwargs, rollout_kwargs, extra_groups_kwargs_list = topology
            extra_groups_kwargs_list = extra_groups_kwargs_list or []
        else:
            raise ValueError("build_topology must return 2 or 3 values")

        if len(extra_groups_kwargs_list) != len(self.extra_groups):
            raise ValueError(
                "build_topology extra_groups kwargs size mismatch: "
                f"expect {len(self.extra_groups)}, got {len(extra_groups_kwargs_list)}"
            )

        for k, v in trainer_kwargs.items():
            assert len(v) == trainer.world_size, f"trainer_kwargs[{k}] must have length of {trainer.world_size}"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] must have length of {rollout.world_size}"
        for i, extra_groups_kwargs in enumerate(extra_groups_kwargs_list):
            for k, v in extra_groups_kwargs.items():
                assert len(v) == self.extra_groups[i].world_size, (
                    f"extra_groups_kwargs[{k}] must have length of {self.extra_groups[i].world_size}"
                )

        trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size
        for i, extra_groups_kwargs in enumerate(extra_groups_kwargs_list):
            extra_groups_kwargs["method"] = ["init_process_group"] * self.extra_groups[i].world_size

        # 3. init process group between all workers
        init_refs = trainer.execute_checkpoint_engine(**trainer_kwargs) + rollout.execute_checkpoint_engine(
            **rollout_kwargs
        )
        for wg, extra_groups_kwargs in zip(self.extra_groups, extra_groups_kwargs_list, strict=True):
            init_refs += wg.execute_checkpoint_engine(**extra_groups_kwargs)
        ray.get(init_refs)

    def add_replicas(self, replicas: list[RolloutReplica]):
        """Add rollout replicas to the manager for elastic scale up, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to add.
        """
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        replicas_set = set(replicas)
        self.replicas = [r for r in self.replicas if r not in replicas_set]

    @auto_await
    async def sleep_replicas(self):
        """Sleep all rollout replicas: free weight and kv_cache device memory."""
        # skip sleep replicas for disaggregated rollout
        if self.backend != "naive":
            return
        await asyncio.gather(*[r.sleep() for r in self.replicas])

    @auto_await
    async def update_weights(self, global_steps: int = None, post_finalize_callback=None):
        """Update weights from trainer to rollout replicas.

        Args:
            global_steps: The global steps of the trainer.
            post_finalize_callback: An optional async callable function.
        """

        # 0. update weights for sync training with colocated trainer and rollout
        if self.backend == "naive":
            ray.get(self.trainer.update_weights(global_steps=global_steps))
            return

        # 1. abort and save all unfinished requests for partial rollout
        await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])

        # 2. create a temporay worker group for all replicas
        workers = []
        for replica in self.replicas:
            workers.extend(replica.workers)
        rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))
        trainer = self.trainer

        # 3. build process group
        self.build_process_group(rollout)

        # + 3.5 wait for any background tasks from the previous update_weights call
        # (e.g. drain_and_load_weights for old_log_prob_server) to complete before
        # transferring new weights.
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks)

        # 4. update weights of all workers
        update_refs = trainer.update_weights(global_steps=global_steps) + rollout.update_weights(
            global_steps=global_steps
        )
        for wg in self.extra_groups:
            update_refs += wg.update_weights()
        ray.get(update_refs)

        # 5. finalize all workers
        finalize_refs = trainer.execute_checkpoint_engine(
            ["finalize"] * trainer.world_size
        ) + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        for wg in self.extra_groups:
            finalize_refs += wg.execute_checkpoint_engine(["finalize"] * wg.world_size)
        ray.get(finalize_refs)

        # + 5.5 fire post-finalize callback, without awaiting it
        # (e.g. drain_and_load_weights for old_log_prob_server)
        if post_finalize_callback is not None:
            self._background_tasks.add(asyncio.create_task(post_finalize_callback()))

        # 6. resume all unfinished requests for partial rollout
        await asyncio.gather(*[r.resume_generation() for r in self.replicas])
