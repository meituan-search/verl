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
Hybrid Elastic Actor Worker for VERL

HybridElasticActorWorker extends ActorRolloutRefWorker to support dynamic
switching between rollout and training roles within the same process group.

Overview
--------
Unlike a static worker that is permanently assigned to either training or
rollout, HybridElasticActorWorker can participate in either role depending
on the current scheduling decision. This enables elastic resource sharing:
workers not needed for training can serve rollout requests, and vice versa.

Resource allocation is controlled at the rank level:
- A rank assigned to ``TRAIN`` participates in the DP training group.
- A rank assigned to ``ROLLOUT`` serves inference requests.
- Roles can be reassigned at any PPO iteration boundary.

Usage
-----
::

    worker = HybridElasticActorWorker(config, role="actor_rollout")
    worker.init_model()

    # Route rank 0-3 to training, rank 4-7 to rollout
    worker.switch_to_train(new_train_world_ranks=[0, 1, 2, 3], param_version=0)
    # Later, reassign
    worker.switch_to_rollout(param_version=1)
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import ray
import torch
import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ElasticMode(Enum):
    """Current role of a hybrid elastic worker."""

    TRAIN = auto()  # Participating in training DP group
    ROLLOUT = auto()  # Serving inference requests
    SWITCHING = auto()  # Transitioning between roles


@dataclass
class ElasticState:
    """Runtime state of a hybrid elastic worker."""

    resource_id: str
    current_mode: ElasticMode = ElasticMode.TRAIN
    param_version: int = -1
    train_world_ranks: list = field(default_factory=list)
    rollout_world_ranks: list = field(default_factory=list)
    last_switch_time: float = 0.0
    total_switches: int = 0
    is_healthy: bool = True

    def record_switch(self, new_mode: ElasticMode) -> None:
        self.current_mode = new_mode
        self.last_switch_time = time.time()
        self.total_switches += 1


class HybridElasticActorWorker(ActorRolloutRefWorker):
    """
    Elastic worker that can dynamically switch between training and rollout.

    On ``init_model``, the actor engine is patched in-place with the
    appropriate elastic mixin (``ElasticFSDPMixin`` or
    ``ElasticMegatronMixin``) so that ``rebuild_dp_group`` is available
    without subclassing the engine.

    Mode transitions
    ----------------
    TRAIN → ROLLOUT:
        1. Offload actor weights to CPU.
        2. Wake up the rollout server.

    ROLLOUT → TRAIN:
        1. Sleep the rollout server (free GPU memory).
        2. Load actor weights back to GPU.
        3. Rebuild the DP communication group.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elastic_state: Optional[ElasticState] = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model and patch engine for elastic DP rebuild."""
        super().init_model()

        if self.actor is not None:
            self._patch_engine_to_elastic()

        self._elastic_state = ElasticState(resource_id=f"rank{dist.get_rank()}")
        logger.info(f"[HybridElasticActorWorker] Initialized, rank={dist.get_rank()}")

    def _patch_engine_to_elastic(self) -> None:
        """
        Patch ``self.actor.engine`` in-place with the elastic mixin so that
        ``engine.rebuild_dp_group`` becomes available.

        The mixin is selected based on the actor strategy in config. If the
        engine already has ``rebuild_dp_group`` (e.g., already patched or an
        elastic subclass), this is a no-op.
        """
        from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls

        engine = self.actor.engine

        if callable(getattr(engine, "rebuild_dp_group", None)):
            logger.debug("[HybridElasticActorWorker] Engine already has rebuild_dp_group, skipping patch")
            return

        strategy = self._get_actor_strategy()
        if strategy is None:
            logger.warning("[HybridElasticActorWorker] Cannot detect actor strategy; engine not patched")
            return

        original_cls = type(engine)
        try:
            elastic_cls = get_elastic_engine_cls(strategy, original_cls)
        except KeyError:
            logger.warning(f"[HybridElasticActorWorker] No elastic mixin registered for strategy={strategy!r}")
            return

        engine.__class__ = elastic_cls
        logger.info(f"[HybridElasticActorWorker] Engine patched: {original_cls.__name__} → {elastic_cls.__name__}")

    def _get_actor_strategy(self) -> Optional[str]:
        """
        Read actor strategy from config.

        Returns the strategy string (e.g. ``"megatron"``, ``"fsdp"``,
        ``"fsdp2"``) or ``None`` if it cannot be determined.
        """
        try:
            return self.config.actor.strategy
        except AttributeError:
            pass

        # Fallback: infer from engine class name
        if self.actor and self.actor.engine:
            cls_name = type(self.actor.engine).__name__.lower()
            if "megatron" in cls_name:
                return "megatron"
            if "fsdp" in cls_name:
                return "fsdp2"

        return None

    # -------------------------------------------------------------------------
    # Mode switching
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_train(self, new_train_world_ranks: list[int], param_version: int) -> bool:
        """
        Switch this worker from rollout to training mode.

        Steps:
        1. Sleep the rollout server to free GPU memory for training.
        2. Load actor weights back to GPU.
        3. Rebuild the DP communication group.

        Args:
            new_train_world_ranks: Global ranks forming the new training DP group.
            param_version: Parameter version to record in elastic state.

        Returns:
            True if the switch succeeded, False otherwise.
        """
        assert self._elastic_state is not None, "call init_model() before switch_to_train()"

        if self._elastic_state.current_mode == ElasticMode.TRAIN:
            logger.debug("[HybridElasticActorWorker] Already in TRAIN mode")
            return True

        logger.info(
            f"[HybridElasticActorWorker rank={dist.get_rank()}] "
            f"ROLLOUT → TRAIN  dp_size={len(new_train_world_ranks)}  param_version={param_version}"
        )

        try:
            self._sleep_rollout()
            self._load_actor_to_gpu()
            self._rebuild_dp_group(new_train_world_ranks)

            self._elastic_state.train_world_ranks = new_train_world_ranks
            self._elastic_state.param_version = param_version
            self._elastic_state.record_switch(ElasticMode.TRAIN)
            return True

        except Exception:
            logger.exception("[HybridElasticActorWorker] switch_to_train failed")
            self._elastic_state.is_healthy = False
            return False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_rollout(self, param_version: int) -> bool:
        """
        Switch this worker from training to rollout mode.

        Steps:
        1. Offload actor weights to CPU to free GPU memory.
        2. Wake up the rollout server.

        Args:
            param_version: Parameter version to record in elastic state.

        Returns:
            True if the switch succeeded, False otherwise.
        """
        assert self._elastic_state is not None, "call init_model() before switch_to_rollout()"

        if self._elastic_state.current_mode == ElasticMode.ROLLOUT:
            logger.debug("[HybridElasticActorWorker] Already in ROLLOUT mode")
            return True

        logger.info(f"[HybridElasticActorWorker rank={dist.get_rank()}] TRAIN → ROLLOUT  param_version={param_version}")

        try:
            self._offload_actor_to_cpu()
            self._wake_up_rollout()

            self._elastic_state.param_version = param_version
            self._elastic_state.record_switch(ElasticMode.ROLLOUT)
            return True

        except Exception:
            logger.exception("[HybridElasticActorWorker] switch_to_rollout failed")
            self._elastic_state.is_healthy = False
            return False

    # -------------------------------------------------------------------------
    # DP group rebuild (pass-through to engine)
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the training DP process group.

        Delegates to ``self.actor.engine.rebuild_dp_group``. All ranks in the
        current global process group must call this simultaneously because
        ``dist.new_group`` is a collective operation.

        Args:
            new_world_ranks: Ordered list of global ranks in the new DP group.
        """
        self._rebuild_dp_group(new_world_ranks)

    # -------------------------------------------------------------------------
    # State introspection
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_global_rank(self) -> int:
        """Return this worker's global rank."""
        return dist.get_rank()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_elastic_state(self) -> Optional[dict]:
        """Return a serializable snapshot of the current elastic state."""
        if self._elastic_state is None:
            return None
        s = self._elastic_state
        return {
            "resource_id": s.resource_id,
            "current_mode": s.current_mode.name,
            "param_version": s.param_version,
            "train_world_ranks": s.train_world_ranks,
            "rollout_world_ranks": s.rollout_world_ranks,
            "last_switch_time": s.last_switch_time,
            "total_switches": s.total_switches,
            "is_healthy": s.is_healthy,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """Delegate DP group rebuild to the (patched) engine."""
        if not callable(getattr(self.actor.engine, "rebuild_dp_group", None)):
            raise AttributeError(
                f"Engine {type(self.actor.engine).__name__} does not implement "
                "rebuild_dp_group(). Ensure the engine was patched via "
                "_patch_engine_to_elastic()."
            )
        self.actor.engine.rebuild_dp_group(new_world_ranks)

    def _sleep_rollout(self) -> None:
        """Put the rollout server to sleep to free GPU memory."""
        if not hasattr(self, "rollout") or self.rollout is None:
            return
        if hasattr(self.rollout, "servers") and self.rollout.servers:
            ray.get([server.sleep.remote() for server in self.rollout.servers])
        elif hasattr(self.rollout, "sleep"):
            self.rollout.sleep()
        torch.cuda.empty_cache()
        gc.collect()

    def _wake_up_rollout(self) -> None:
        """Resume the rollout server for inference."""
        if not hasattr(self, "rollout") or self.rollout is None:
            return
        if hasattr(self.rollout, "servers") and self.rollout.servers:
            ray.get([server.wake_up.remote() for server in self.rollout.servers])
        elif hasattr(self.rollout, "wake_up"):
            self.rollout.wake_up()

    def _offload_actor_to_cpu(self) -> None:
        """Move actor model weights to CPU, keeping optimizer state on CPU too."""
        if not hasattr(self, "actor") or self.actor is None:
            return
        self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        torch.cuda.empty_cache()
        gc.collect()

    def _load_actor_to_gpu(self) -> None:
        """Restore actor model weights from CPU back to GPU."""
        if not hasattr(self, "actor") or self.actor is None:
            return
        self.actor.engine.to("device", model=True, optimizer=False, grad=False)
