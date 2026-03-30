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

"""
Elastic Worker for VERL

ElasticActorRolloutRefWorker extends ActorRolloutRefWorker to support dynamic
switching between rollout and training modes. This is the core building block
for elastic scheduling.

Key capabilities:
1. Dual-mode initialization: both actor engine and rollout engine are initialized
2. Mode switching via sleep/wake_up mechanism
3. DP group rebuild support for FSDP2 and Megatron after mode switch
4. Parameter synchronization hooks for elastic resource joining
"""

import asyncio
import gc
import logging
import time
from enum import Enum, auto
from typing import Optional

import torch
import torch.distributed as dist

from verl.experimental.elastic_scheduling.engine.fsdp import FSDP2DPRebuildManager
from verl.experimental.elastic_scheduling.engine.megatron import MegatronDPRebuildManager

logger = logging.getLogger(__name__)


class ElasticMode(Enum):
    """Operation mode of an elastic worker."""

    ROLLOUT = auto()  # Worker is acting as a rollout server
    TRAIN = auto()  # Worker is participating in training
    SWITCHING = auto()  # Worker is in the middle of a mode switch


class ElasticWorkerState:
    """State tracking for an elastic worker."""

    def __init__(self, resource_id: str, initial_mode: ElasticMode = ElasticMode.ROLLOUT):
        self.resource_id = resource_id
        self.current_mode = initial_mode
        self.param_version: int = -1  # -1 means not yet synced
        self.last_switch_time: float = 0.0
        self.total_switches: int = 0
        self.is_healthy: bool = True

    def record_switch(self, new_mode: ElasticMode):
        self.current_mode = new_mode
        self.last_switch_time = time.time()
        self.total_switches += 1

    def update_param_version(self, version: int):
        self.param_version = version


class ElasticWorkerMixin:
    """
    Mixin that adds elastic scheduling capabilities to ActorRolloutRefWorker.

    This mixin provides:
    1. Mode switching (Rollout <-> Train)
    2. DP group rebuild for FSDP2 and Megatron
    3. Parameter synchronization hooks

    Usage:
        class MyElasticWorker(ElasticWorkerMixin, ActorRolloutRefWorker):
            pass

    The mixin assumes the base class has:
    - self.actor: training engine
    - self.rollout: rollout engine
    - self.config: configuration
    """

    def init_elastic_state(self, resource_id: str, initial_mode: ElasticMode = ElasticMode.ROLLOUT):
        """
        Initialize elastic state. Must be called after __init__.

        Args:
            resource_id: Unique identifier for this elastic resource
            initial_mode: Initial mode (default: ROLLOUT)
        """
        self.elastic_state = ElasticWorkerState(resource_id, initial_mode)
        self._fsdp2_rebuild_manager: Optional[FSDP2DPRebuildManager] = None
        self._megatron_rebuild_manager: Optional[MegatronDPRebuildManager] = None
        self._mode_switch_lock = asyncio.Lock()

        # Initialize DP rebuild manager based on strategy
        strategy = getattr(self.config, "actor", {})
        if hasattr(strategy, "strategy"):
            actor_strategy = strategy.strategy
        else:
            actor_strategy = getattr(strategy, "get", lambda k, d: d)("strategy", "fsdp2")

        self._actor_strategy = actor_strategy
        logger.info(f"[ElasticWorker] {resource_id} initialized in {initial_mode.name} mode, strategy={actor_strategy}")

    async def switch_to_train(
        self,
        new_train_world_ranks: list[int],
        param_version: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> bool:
        """
        Switch this elastic worker from Rollout mode to Train mode.

        Steps:
        1. Stop accepting new rollout requests
        2. Wait for in-flight rollout requests to complete
        3. Sleep rollout engine (release KV cache and inference weights)
        4. Rebuild DP communication group with new training workers
        5. Update param version tracking

        Args:
            new_train_world_ranks: Global ranks of all workers in the new train DP group
            param_version: Current parameter version from trainer
            tensor_parallel_size: TP size (fixed)
            pipeline_parallel_size: PP size (fixed)

        Returns:
            True if switch succeeded, False otherwise
        """
        async with self._mode_switch_lock:
            if self.elastic_state.current_mode == ElasticMode.TRAIN:
                logger.warning(f"[ElasticWorker] {self.elastic_state.resource_id} already in TRAIN mode")
                return True

            logger.info(
                f"[ElasticWorker] {self.elastic_state.resource_id}: "
                f"Switching ROLLOUT -> TRAIN, param_version={param_version}"
            )
            self.elastic_state.current_mode = ElasticMode.SWITCHING

            try:
                # Step 1: Sleep rollout engine to free GPU memory
                await self._sleep_rollout_for_training()

                # Step 2: Rebuild DP group for training
                await self._rebuild_dp_for_training(
                    new_train_world_ranks=new_train_world_ranks,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                )

                # Step 3: Update state
                self.elastic_state.record_switch(ElasticMode.TRAIN)
                self.elastic_state.update_param_version(param_version)

                logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Successfully switched to TRAIN mode")
                return True

            except Exception as e:
                logger.error(f"[ElasticWorker] {self.elastic_state.resource_id}: Failed to switch to TRAIN: {e}")
                self.elastic_state.current_mode = ElasticMode.ROLLOUT  # Rollback
                self.elastic_state.is_healthy = False
                return False

    async def switch_to_rollout(
        self,
        new_rollout_world_ranks: list[int],
        param_version: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> bool:
        """
        Switch this elastic worker from Train mode to Rollout mode.

        Steps:
        1. Finish current training step (if any)
        2. Save model state (optimizer state stays in actor engine)
        3. Wake up rollout engine
        4. Sync latest parameters to rollout engine
        5. Update rollout DP group

        Args:
            new_rollout_world_ranks: Global ranks for the rollout inference group
            param_version: Current parameter version to sync
            tensor_parallel_size: TP size for rollout
            pipeline_parallel_size: PP size for rollout

        Returns:
            True if switch succeeded, False otherwise
        """
        async with self._mode_switch_lock:
            if self.elastic_state.current_mode == ElasticMode.ROLLOUT:
                logger.warning(f"[ElasticWorker] {self.elastic_state.resource_id} already in ROLLOUT mode")
                return True

            logger.info(
                f"[ElasticWorker] {self.elastic_state.resource_id}: "
                f"Switching TRAIN -> ROLLOUT, param_version={param_version}"
            )
            self.elastic_state.current_mode = ElasticMode.SWITCHING

            try:
                # Step 1: Offload actor engine to CPU (keep params in memory)
                await self._offload_actor_for_rollout()

                # Step 2: Wake up rollout engine
                await self._wake_up_rollout_for_inference(new_rollout_world_ranks)

                # Step 3: Update state (params will be synced by ElasticParamSync)
                self.elastic_state.record_switch(ElasticMode.ROLLOUT)
                self.elastic_state.update_param_version(param_version)

                logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Successfully switched to ROLLOUT mode")
                return True

            except Exception as e:
                logger.error(f"[ElasticWorker] {self.elastic_state.resource_id}: Failed to switch to ROLLOUT: {e}")
                self.elastic_state.current_mode = ElasticMode.TRAIN  # Rollback
                self.elastic_state.is_healthy = False
                return False

    async def _sleep_rollout_for_training(self):
        """
        Sleep the rollout engine to free GPU memory for training.

        Uses the existing sleep/wake_up mechanism from checkpoint_engine.
        """
        if not hasattr(self, "rollout") or self.rollout is None:
            logger.debug("[ElasticWorker] No rollout engine to sleep")
            return

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Sleeping rollout engine...")

        # Level 2: fully release weights and KV cache
        if hasattr(self.rollout, "sleep"):
            await self.rollout.sleep(level=2)
        elif hasattr(self.rollout, "_sleep_hybrid"):
            await self.rollout._sleep_hybrid()
        else:
            logger.warning("[ElasticWorker] Rollout engine has no sleep() method")

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Rollout engine sleeping")

    async def _wake_up_rollout_for_inference(self, world_ranks: list[int]):
        """
        Wake up the rollout engine for inference.

        After waking up, the rollout engine will accept new requests.
        Parameters need to be synced separately via ElasticParamSync.
        """
        if not hasattr(self, "rollout") or self.rollout is None:
            logger.debug("[ElasticWorker] No rollout engine to wake up")
            return

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Waking up rollout engine...")

        if hasattr(self.rollout, "wake_up"):
            await self.rollout.wake_up()
        elif hasattr(self.rollout, "_wake_up_hybrid"):
            await self.rollout._wake_up_hybrid()
        else:
            logger.warning("[ElasticWorker] Rollout engine has no wake_up() method")

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Rollout engine awake")

    async def _offload_actor_for_rollout(self):
        """
        Offload actor engine to CPU to free GPU memory for rollout.

        The actor's parameters stay in CPU memory (not disk) so they can be
        quickly restored when switching back to training mode.
        """
        if not hasattr(self, "actor") or self.actor is None:
            logger.debug("[ElasticWorker] No actor engine to offload")
            return

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Offloading actor to CPU...")

        try:
            # Try FSDP2 offload
            from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu, offload_fsdp_optimizer

            if hasattr(self.actor, "engine") and hasattr(self.actor.engine, "module"):
                offload_fsdp_model_to_cpu(self.actor.engine.module)
            elif hasattr(self.actor, "actor_module"):
                offload_fsdp_model_to_cpu(self.actor.actor_module)

            if hasattr(self.actor, "optimizer"):
                offload_fsdp_optimizer(self.actor.optimizer)

        except ImportError:
            # Fallback: manual CPU offload
            if hasattr(self.actor, "engine"):
                for param in self.actor.engine.parameters():
                    param.data = param.data.cpu()
                    if param.grad is not None:
                        param.grad = param.grad.cpu()

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Actor offloaded to CPU")

    async def _rebuild_dp_for_training(
        self,
        new_train_world_ranks: list[int],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """
        Rebuild DP communication group for training.

        Routes to FSDP2 or Megatron rebuild based on actor strategy.
        """
        strategy = self._actor_strategy

        if strategy in ("fsdp2", "fsdp"):
            await self._rebuild_fsdp2_dp(new_train_world_ranks, tensor_parallel_size, pipeline_parallel_size)
        elif strategy == "megatron":
            await self._rebuild_megatron_dp(new_train_world_ranks, tensor_parallel_size, pipeline_parallel_size)
        else:
            logger.warning(f"[ElasticWorker] Unknown actor strategy '{strategy}', skipping DP rebuild")

    async def _rebuild_fsdp2_dp(
        self,
        new_world_ranks: list[int],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """Rebuild FSDP2 DP group."""
        model = getattr(getattr(self, "actor", None), "engine", None)
        if model is None:
            model = getattr(getattr(self, "actor", None), "actor_module", None)

        optimizer = getattr(getattr(self, "actor", None), "optimizer", None)

        if model is None:
            logger.warning("[ElasticWorker] Could not find actor model for FSDP2 rebuild")
            return

        if self._fsdp2_rebuild_manager is None:
            self._fsdp2_rebuild_manager = FSDP2DPRebuildManager(
                model=model,
                optimizer=optimizer,
                tp_size=tensor_parallel_size,
                pp_size=pipeline_parallel_size,
            )

        global_rank = dist.get_rank()
        self._fsdp2_rebuild_manager.rebuild(new_world_ranks, global_rank)

    async def _rebuild_megatron_dp(
        self,
        new_world_ranks: list[int],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """Rebuild Megatron DP group."""
        model = getattr(getattr(self, "actor", None), "actor_module", None)
        optimizer = getattr(getattr(self, "actor", None), "optimizer", None)

        if model is None:
            logger.warning("[ElasticWorker] Could not find actor model for Megatron rebuild")
            return

        if self._megatron_rebuild_manager is None:
            self._megatron_rebuild_manager = MegatronDPRebuildManager(
                model=model,
                optimizer=optimizer,
            )

        # Identify newly joining ranks
        try:
            from megatron.core import parallel_state

            current_dp_group = parallel_state.get_data_parallel_group()
            current_dp_ranks = list(dist.get_process_group_ranks(current_dp_group))
        except Exception:
            current_dp_ranks = []

        new_member_ranks = [r for r in new_world_ranks if r not in current_dp_ranks]

        new_world_size = len(new_world_ranks)
        self._megatron_rebuild_manager.rebuild(
            new_world_size=new_world_size,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            new_member_ranks=new_member_ranks if new_member_ranks else None,
        )

    def get_elastic_state(self) -> dict:
        """Get current elastic state as a dictionary."""
        return {
            "resource_id": self.elastic_state.resource_id,
            "current_mode": self.elastic_state.current_mode.name,
            "param_version": self.elastic_state.param_version,
            "last_switch_time": self.elastic_state.last_switch_time,
            "total_switches": self.elastic_state.total_switches,
            "is_healthy": self.elastic_state.is_healthy,
        }
