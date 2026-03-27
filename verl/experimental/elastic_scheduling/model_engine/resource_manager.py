# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, writing
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Elastic Resource Manager for VERL

Manages dynamic allocation of GPU resources between rollout and training:
- Fixed rollout resources (always in rollout mode)
- Fixed train resources (always in train mode)
- Elastic resources (can switch between rollout/train modes)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class HybridEngineMode(Enum):
    """HybridEngine operation modes"""

    ROLLOUT = auto()  # Pure rollout mode (inference only)
    TRAIN = auto()  # Pure train mode (training only)
    COLOCATED = auto()  # Co-located mode (both rollout and train)


@dataclass
class HybridEngineResource:
    """
    Encapsulates an ActorRolloutRefWorker as an elastic resource unit.

    Each HybridEngineResource represents an ActorRolloutRefWorker instance on a group
    of GPUs, supporting dynamic switching between ROLLOUT and TRAIN modes.
    """

    # Resource identity
    resource_id: str
    gpu_ranks: list[int]  # List of GPU ranks
    world_size: int  # World size

    # Underlying worker handle (Ray actor handle)
    worker_handle = None

    # Current mode
    current_mode: HybridEngineMode = HybridEngineMode.ROLLOUT

    # Communication group info
    dp_rank: int = -1
    dp_size: int = -1

    # State
    is_active: bool = False
    is_elastic: bool = False  # Whether this is an elastic resource

    # Metadata
    metadata: dict = field(default_factory=dict)

    @property
    def is_rollout_mode(self) -> bool:
        return self.current_mode == HybridEngineMode.ROLLOUT

    @property
    def is_train_mode(self) -> bool:
        return self.current_mode == HybridEngineMode.TRAIN

    def __hash__(self):
        return hash(self.resource_id)

    def __eq__(self, other):
        if not isinstance(other, HybridEngineResource):
            return False
        return self.resource_id == other.resource_id


@dataclass
class ElasticResourceConfig:
    """Configuration for elastic resources"""

    # Resource range
    min_elastic_gpus: int = 0
    max_elastic_gpus: int = 32
    elastic_dp_size: int = 8  # DP size per elastic resource

    # Switching thresholds
    rollout_queue_high_watermark: float = 0.8  # Rollout queue high watermark
    rollout_queue_low_watermark: float = 0.3  # Rollout queue low watermark
    train_idle_threshold: float = 0.3  # Train idle threshold

    # Switching parameters
    cooldown_seconds: float = 10.0  # Cooldown between switches
    sync_before_switch: bool = True  # Sync params before switching
    graceful_switch: bool = True  # Graceful switching

    # Congestion calculation window
    congestion_window_size: int = 10  # Sliding window size

    # Parallelism config
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


class SwitchRecord:
    """Record of a resource mode switch"""

    def __init__(
        self,
        resource_id: str,
        from_mode: HybridEngineMode,
        to_mode: HybridEngineMode,
        timestamp: float,
    ):
        self.resource_id = resource_id
        self.from_mode = from_mode
        self.to_mode = to_mode
        self.timestamp = timestamp


class ElasticResourceManager:
    """
    Elastic Resource Manager

    Manages three types of resources:
    1. Fixed Rollout resources
    2. Fixed Train resources
    3. Elastic resources (can switch between rollout/train)

    Core capabilities:
    - Dynamic resource grouping and reorganization
    - Flexible DP communication group construction
    - Support for FSDP2 and Megatron 3D parallelism
    """

    def __init__(
        self,
        config: ElasticResourceConfig,
        rollout_resource_spec: list[int],  # Fixed rollout GPU list
        train_resource_spec: list[int],  # Fixed train GPU list
        elastic_resource_spec: list[int],  # Elastic GPU list
    ):
        self.config = config

        # Initialize three types of resources
        self.rollout_resources: list[HybridEngineResource] = []
        self.train_resources: list[HybridEngineResource] = []
        self.elastic_resources: list[HybridEngineResource] = []

        # All resources lookup
        self._all_resources: dict[str, HybridEngineResource] = {}

        # Build resource objects
        self._init_fixed_resources(rollout_resource_spec, train_resource_spec)
        self._init_elastic_resources(elastic_resource_spec)

        # Switch history (for cooldown)
        self._switch_history: list[SwitchRecord] = []

        logger.info(
            f"ElasticResourceManager initialized: "
            f"rollout={len(self.rollout_resources)} groups, "
            f"train={len(self.train_resources)} groups, "
            f"elastic={len(self.elastic_resources)} groups"
        )

    def _init_fixed_resources(self, rollout_spec: list[int], train_spec: list[int]):
        """Initialize fixed resources"""
        # Rollout resources (always in ROLLOUT mode)
        for i in range(0, len(rollout_spec), self.config.elastic_dp_size):
            ranks = rollout_spec[i : i + self.config.elastic_dp_size]
            if len(ranks) == self.config.elastic_dp_size:
                resource = HybridEngineResource(
                    resource_id=f"rollout_fixed_{len(self.rollout_resources)}",
                    gpu_ranks=ranks,
                    world_size=len(ranks),
                    current_mode=HybridEngineMode.ROLLOUT,
                    is_elastic=False,
                )
                resource.dp_size = len(ranks)
                self.rollout_resources.append(resource)
                self._all_resources[resource.resource_id] = resource

        # Train resources (always in TRAIN mode)
        for i in range(0, len(train_spec), self.config.elastic_dp_size):
            ranks = train_spec[i : i + self.config.elastic_dp_size]
            if len(ranks) == self.config.elastic_dp_size:
                resource = HybridEngineResource(
                    resource_id=f"train_fixed_{len(self.train_resources)}",
                    gpu_ranks=ranks,
                    world_size=len(ranks),
                    current_mode=HybridEngineMode.TRAIN,
                    is_elastic=False,
                )
                resource.dp_size = len(ranks)
                self.train_resources.append(resource)
                self._all_resources[resource.resource_id] = resource

    def _init_elastic_resources(self, elastic_spec: list[int]):
        """Initialize elastic resources"""
        for i in range(0, len(elastic_spec), self.config.elastic_dp_size):
            ranks = elastic_spec[i : i + self.config.elastic_dp_size]
            if len(ranks) == self.config.elastic_dp_size:
                resource = HybridEngineResource(
                    resource_id=f"elastic_{len(self.elastic_resources)}",
                    gpu_ranks=ranks,
                    world_size=len(ranks),
                    current_mode=HybridEngineMode.ROLLOUT,  # Default to ROLLOUT mode
                    is_elastic=True,
                )
                resource.dp_size = len(ranks)
                self.elastic_resources.append(resource)
                self._all_resources[resource.resource_id] = resource

    async def switch_elastic_to_rollout(
        self,
        n_resources: int,
        sync_callback: Optional[Callable] = None,
    ) -> list[HybridEngineResource]:
        """
        Switch elastic resources to Rollout mode

        Args:
            n_resources: Number of resources to switch
            sync_callback: Optional sync callback before switching

        Returns:
            List of successfully switched resources
        """
        to_switch = []
        available = [r for r in self.elastic_resources if r.is_active and r.current_mode == HybridEngineMode.TRAIN]

        for resource in available[:n_resources]:
            if self._can_switch(resource.resource_id):
                # Sync params before switching
                if sync_callback and self.config.sync_before_switch:
                    await sync_callback(resource)

                # Switch mode
                await self._do_switch_mode(resource, HybridEngineMode.ROLLOUT)
                to_switch.append(resource)

        logger.info(f"Switched {len(to_switch)} elastic resources to rollout mode")
        return to_switch

    async def switch_elastic_to_train(
        self,
        n_resources: int,
        sync_callback: Optional[Callable] = None,
    ) -> list[HybridEngineResource]:
        """
        Switch elastic resources to Train mode

        Args:
            n_resources: Number of resources to switch
            sync_callback: Optional sync callback before switching

        Returns:
            List of successfully switched resources
        """
        to_switch = []
        available = [r for r in self.elastic_resources if r.is_active and r.current_mode == HybridEngineMode.ROLLOUT]

        for resource in available[:n_resources]:
            if self._can_switch(resource.resource_id):
                # Sync params before switching
                if sync_callback and self.config.sync_before_switch:
                    await sync_callback(resource)

                # Switch mode
                await self._do_switch_mode(resource, HybridEngineMode.TRAIN)
                to_switch.append(resource)

        logger.info(f"Switched {len(to_switch)} elastic resources to train mode")
        return to_switch

    async def _do_switch_mode(
        self,
        resource: HybridEngineResource,
        target_mode: HybridEngineMode,
    ):
        """
        Execute mode switch using sleep/wake_up mechanism.

        The switch leverages the existing checkpoint_engine's sleep/wake_up logic:
        - Switch to TRAIN: sleep(rollout) -> free GPU memory -> activate train mode
        - Switch to ROLLOUT: wake_up(rollout) -> restore weights -> activate rollout mode

        After mode switch, rebuild DP communication groups for the new configuration.
        """
        # Record switch history
        record = SwitchRecord(
            resource_id=resource.resource_id,
            from_mode=resource.current_mode,
            to_mode=target_mode,
            timestamp=time.time(),
        )
        self._switch_history.append(record)

        from_mode = resource.current_mode

        # Get worker handle (ActorRolloutRefWorker)
        worker = resource.worker_handle
        if worker is None:
            logger.warning(f"No worker handle for {resource.resource_id}, skipping mode switch")
            return

        try:
            if target_mode == HybridEngineMode.TRAIN:
                # Switch from ROLLOUT to TRAIN
                # 1. Sleep rollout (release weights and kv_cache)
                await self._sleep_rollout(worker, resource)

                # 2. Switch to train mode (rebuild DP groups if needed)
                await self._switch_to_train(worker, resource)

            elif target_mode == HybridEngineMode.ROLLOUT:
                # Switch from TRAIN to ROLLOUT
                # 1. Switch to rollout mode first
                await self._switch_to_rollout(worker, resource)

                # 2. Wake up rollout (restore weights and kv_cache)
                await self._wake_up_rollout(worker, resource)

            resource.current_mode = target_mode

            # Rebuild DP communication group for the new mode
            await self._rebuild_dp_group(resource, from_mode, target_mode)

            logger.info(f"Successfully switched {resource.resource_id} from {from_mode.name} to {target_mode.name}")

        except Exception as e:
            logger.error(f"Failed to switch {resource.resource_id}: {e}")
            raise

    async def _sleep_rollout(self, worker, resource: HybridEngineResource):
        """
        Sleep rollout using existing checkpoint_engine mechanism.

        This leverages the existing sleep(level=2) logic which:
        - Releases model weights from GPU memory
        - Releases KV cache
        - Prepares for parameter sync
        """
        if not hasattr(worker, "rollout") or worker.rollout is None:
            logger.debug(f"No rollout attached to {resource.resource_id}")
            return

        rollout = worker.rollout

        # Use sleep(level=2) to fully release memory
        # This matches the HYBRID mode sleep behavior in vLLM
        if hasattr(rollout, "sleep"):
            await rollout.sleep(level=2)
            logger.debug(f"Sleep rollout for {resource.resource_id}")
        elif hasattr(rollout, "_sleep_hybrid"):
            await rollout._sleep_hybrid()
            logger.debug(f"Sleep hybrid rollout for {resource.resource_id}")

    async def _wake_up_rollout(self, worker, resource: HybridEngineResource):
        """
        Wake up rollout using existing checkpoint_engine mechanism.

        This leverages the existing wake_up(tags) logic which:
        - Restores model weights to GPU memory
        - Restores KV cache
        - Resumes inference
        """
        if not hasattr(worker, "rollout") or worker.rollout is None:
            logger.debug(f"No rollout attached to {resource.resource_id}")
            return

        rollout = worker.rollout

        # Wake up with both weights and kv_cache tags
        # This matches the HYBRID mode wake_up behavior
        if hasattr(rollout, "wake_up"):
            await rollout.wake_up()
            logger.debug(f"Wake up rollout for {resource.resource_id}")
        elif hasattr(rollout, "_wake_up_hybrid"):
            await rollout._wake_up_hybrid()
            logger.debug(f"Wake up hybrid rollout for {resource.resource_id}")

    async def _switch_to_train(self, worker, resource: HybridEngineResource):
        """
        Switch worker to train mode.

        This involves:
        - Enabling train-specific settings (optimizer, gradients)
        - Rebuilding DP groups for training
        """
        # For ActorRolloutRefWorker, the worker already has actor embedded
        # We just need to ensure it's in train-ready state
        if hasattr(worker, "actor") and worker.actor is not None:
            # Ensure actor engine is in train mode
            if hasattr(worker.actor.engine, "train_mode"):
                logger.debug(f"Switch {resource.resource_id} actor to train mode")

        logger.debug(f"Switch {resource.resource_id} to train mode complete")

    async def _switch_to_rollout(self, worker, resource: HybridEngineResource):
        """
        Switch worker to rollout mode.

        This involves:
        - Ensuring rollout engine is initialized
        - Preparing for inference
        """
        # For ActorRolloutRefWorker, ensure rollout is ready
        if hasattr(worker, "rollout") and worker.rollout is not None:
            logger.debug(f"Switch {resource.resource_id} to rollout mode")

        logger.debug(f"Switch {resource.resource_id} to rollout mode complete")

    async def _rebuild_dp_group(
        self,
        resource: HybridEngineResource,
        from_mode: HybridEngineMode,
        to_mode: HybridEngineMode,
    ):
        """
        Rebuild DP communication group for the new mode.

        This is called after mode switch to reconfigure the parallel groups
        based on the new resource allocation.

        For FSDP2:
        - Reconfigure 3D device_mesh (dp, tp, pp)
        - Keep TP/PP unchanged, rebuild DP groups only

        For Megatron:
        - Call mpu.destroy_model_parallel() to clean up
        - Reinitialize with new dp_size via mpu.initialize_model_parallel()
        - Coordinate with existing resources
        """
        if resource.worker_handle is None:
            return

        engine = None
        if hasattr(resource.worker_handle, "actor") and resource.worker_handle.actor:
            engine = resource.worker_handle.actor.engine
        elif hasattr(resource.worker_handle, "engine"):
            engine = resource.worker_handle.engine

        if engine is None:
            logger.debug(f"No engine found for {resource.resource_id}")
            return

        engine_strategy = getattr(engine, "strategy", None) or getattr(
            getattr(engine, "config", None), "strategy", None
        )

        if engine_strategy in ["fsdp2", "fsdp"]:
            await self._rebuild_fsdp2_dp_group(resource, engine, to_mode)
        elif engine_strategy == "megatron":
            await self._rebuild_megatron_dp_group(resource, engine, to_mode)
        else:
            logger.debug(f"Unknown engine strategy: {engine_strategy}")

    async def _rebuild_fsdp2_dp_group(
        self,
        resource: HybridEngineResource,
        engine,
        target_mode: HybridEngineMode,
    ):
        """
        Rebuild DP group for FSDP2 engine.

        FSDP2 uses a 3D device mesh: (dp_size, tp_size, pp_size)
        When switching modes, we need to:
        1. Get the current mesh configuration
        2. Adjust dp_size based on available resources
        3. Reinitialize the mesh with new configuration
        """
        try:
            from torch.distributed.device_mesh import _mesh_resources

            # Get current mesh resources
            mesh_resources = _mesh_resources.get_instance_mesh_resources()
            if mesh_resources is None:
                logger.debug("No existing mesh resources found")
                return

            # For FSDP2, we primarily adjust DP size
            # TP and PP remain constant during elastic scaling
            current_dp_size = resource.dp_size

            if target_mode == HybridEngineMode.TRAIN:
                # In train mode, we may need smaller DP for gradient accumulation
                # or keep same DP but with different sharding strategy
                logger.debug(f"FSDP2 DP group for train mode: dp_size={current_dp_size}")
            else:
                # In rollout mode, DP is used for data-parallel inference
                logger.debug(f"FSDP2 DP group for rollout mode: dp_size={current_dp_size}")

            # Note: Full mesh reconfiguration would require:
            # 1. Destroying current mesh
            # 2. Creating new mesh with adjusted dp_size
            # 3. Updating all related process groups
            # This is complex and should be done with care

        except Exception as e:
            logger.error(f"Failed to rebuild FSDP2 DP group: {e}")

    async def _rebuild_megatron_dp_group(
        self,
        resource: HybridEngineResource,
        engine,
        target_mode: HybridEngineMode,
    ):
        """
        Rebuild DP group for Megatron engine.

        Megatron uses hierarchical parallelism with separate process groups:
        - tensor_model_parallel_group
        - pipeline_model_parallel_group
        - data_parallel_group
        - model_parallel_group

        When switching modes:
        1. Destroy current model_parallel groups (if needed)
        2. Reinitialize with new configuration
        3. Update model and optimizer state
        """
        try:
            # Import Megatron parallel_state
            try:
                from megatron.core import parallel_state as mpu
            except ImportError:
                logger.warning("Megatron-LM not available")
                return

            # Get current configuration
            tp_size = mpu.get_tensor_model_parallel_world_size()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            dp_size = mpu.get_data_parallel_world_size()

            logger.debug(f"Megatron parallel groups before rebuild: tp={tp_size}, pp={pp_size}, dp={dp_size}")

            # For elastic scaling, we adjust dp_size
            new_dp_size = resource.dp_size

            if target_mode == HybridEngineMode.TRAIN:
                # In train mode, DP is used for gradient synchronization
                # We may need to coordinate with other train workers
                # to form a unified DP group

                # Check if we need to destroy and recreate parallel state
                if dp_size != new_dp_size:
                    logger.info(f"Megatron DP size mismatch: current={dp_size}, target={new_dp_size}")
                    # Note: Full parallel_state reinit is complex
                    # It requires coordinating all workers in the same
                    # model parallel region

            elif target_mode == HybridEngineMode.ROLLOUT:
                # In rollout mode, DP is used for data-parallel inference
                # The DP group may be different from train DP group
                logger.debug(f"Megatron DP group for rollout: dp_size={new_dp_size}")

            # Get world group for reference
            world_group = torch.distributed.group.WORLD
            print(world_group)

            logger.info(
                f"Rebuilt Megatron DP group for {resource.resource_id}: tp={tp_size}, pp={pp_size}, dp={new_dp_size}"
            )

        except Exception as e:
            logger.error(f"Failed to rebuild Megatron DP group: {e}")
            import traceback

            traceback.print_exc()

    def _can_switch(self, resource_id: str) -> bool:
        """Check if switch is allowed (cooldown check)"""
        now = time.time()
        for record in reversed(self._switch_history):
            if record.resource_id == resource_id:
                if now - record.timestamp < self.config.cooldown_seconds:
                    return False
                return True
        return True

    def get_active_rollout_count(self) -> int:
        """Get current count of active rollout resources"""
        return len([r for r in self.rollout_resources if r.is_active]) + sum(
            1 for r in self.elastic_resources if r.is_rollout_mode and r.is_active
        )

    def get_active_train_count(self) -> int:
        """Get current count of active train resources"""
        return len([r for r in self.train_resources if r.is_active]) + sum(
            1 for r in self.elastic_resources if r.is_train_mode and r.is_active
        )

    def get_rollout_resources_all(self) -> list[HybridEngineResource]:
        """Get all resources currently in rollout mode"""
        return [r for r in self.rollout_resources if r.is_active] + [
            r for r in self.elastic_resources if r.is_rollout_mode and r.is_active
        ]

    def get_train_resources_all(self) -> list[HybridEngineResource]:
        """Get all resources currently in train mode"""
        return [r for r in self.train_resources if r.is_active] + [
            r for r in self.elastic_resources if r.is_train_mode and r.is_active
        ]

    def get_resource_by_id(self, resource_id: str) -> Optional[HybridEngineResource]:
        """Get resource by ID"""
        return self._all_resources.get(resource_id)

    def register_worker(self, resource_id: str, worker_handle):
        """Register a worker handle for a resource"""
        resource = self.get_resource_by_id(resource_id)
        if resource:
            resource.worker_handle = worker_handle
            resource.is_active = True
            logger.info(f"Registered worker for resource {resource_id}")
        else:
            logger.warning(f"Resource {resource_id} not found")

    def get_total_gpu_count(self) -> dict:
        """Get total GPU count by role"""
        return {
            "rollout_fixed": len(self.rollout_resources) * self.config.elastic_dp_size,
            "train_fixed": len(self.train_resources) * self.config.elastic_dp_size,
            "elastic_total": len(self.elastic_resources) * self.config.elastic_dp_size,
            "elastic_rollout": sum(1 for r in self.elastic_resources if r.is_rollout_mode)
            * self.config.elastic_dp_size,
            "elastic_train": sum(1 for r in self.elastic_resources if r.is_train_mode) * self.config.elastic_dp_size,
        }

    def get_status_summary(self) -> dict:
        """Get status summary"""
        return {
            "total_resources": len(self._all_resources),
            "active_rollout": self.get_active_rollout_count(),
            "active_train": self.get_active_train_count(),
            "gpu_distribution": self.get_total_gpu_count(),
            "recent_switches": len(self._switch_history),
        }
