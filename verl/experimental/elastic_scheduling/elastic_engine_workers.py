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
Elastic Actor Worker for VERL

ElasticActorWorker extends ActorRolloutRefWorker to support dynamic switching
of the **training engine** between active-training and offloaded states.

Design principle
----------------
This class is **training-engine-only**:

- ``switch_to_train``   – load actor weights back to GPU and rebuild DP group.
- ``switch_to_rollout`` – offload actor weights to CPU (free GPU for rollout server).
- ``rebuild_dp_group``  – rebuild the DP communication group (FSDP2 / Megatron).

All rollout server lifecycle (wake_up / sleep / abort) is managed exclusively
by ``ElasticAgentLoopManager``.  ElasticTrainer calls into the rollouter to
trigger those operations; ElasticActorWorker never touches the rollout server.

Role transitions
----------------
TRAIN → ROLLOUT:
    1. Offload actor model weights to CPU  (frees GPU for rollout server).
    (Rollout server wake_up is done by ElasticAgentLoopManager via rollouter.)

ROLLOUT → TRAIN:
    1. Load actor model weights back to GPU.
    2. Rebuild DP communication group.
    (Rollout server sleep is done by ElasticAgentLoopManager via rollouter
    *before* this call, so GPU memory is already available.)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ElasticMode(Enum):
    """Current role of an elastic actor worker."""

    TRAIN = auto()  # Participating in training DP group
    ROLLOUT = auto()  # Actor weights offloaded; GPU used by rollout server
    SWITCHING = auto()


@dataclass
class ElasticState:
    """Runtime state of an elastic actor worker."""

    resource_id: str
    current_mode: ElasticMode = ElasticMode.TRAIN
    param_version: int = -1
    train_world_ranks: list = field(default_factory=list)
    last_switch_time: float = 0.0
    total_switches: int = 0
    is_healthy: bool = True

    def record_switch(self, new_mode: ElasticMode) -> None:
        self.current_mode = new_mode
        self.last_switch_time = time.time()
        self.total_switches += 1


class ElasticActorWorker(ActorRolloutRefWorker):
    """
    Elastic actor worker that manages **training engine** state only.

    Rollout server lifecycle (wake_up / sleep / abort_all_requests) is
    handled entirely by ``ElasticAgentLoopManager`` and is NOT the
    responsibility of this class.

    The typical switch sequence orchestrated by ``ElasticCoordinator`` is:

    TRAIN → ROLLOUT
    ~~~~~~~~~~~~~~~
    1. [ElasticActorWorker]   offload actor weights to CPU
                              (``switch_to_rollout``)
    2. [ElasticAgentLoopManager via rollouter]  wake_up rollout server
                              (``add_elastic_replica``)

    ROLLOUT → TRAIN
    ~~~~~~~~~~~~~~~
    1. [ElasticAgentLoopManager via rollouter]  sleep rollout server + abort
                              (``remove_elastic_replica``)
    2. [ElasticActorWorker]   load actor weights to GPU + rebuild DP group
                              (``switch_to_train``)
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
        logger.info(f"[ElasticActorWorker] Initialized, rank={dist.get_rank()}")

    def _patch_engine_to_elastic(self) -> None:
        """
        Patch ``self.actor.engine`` in-place with the elastic mixin so that
        ``engine.rebuild_dp_group`` becomes available.

        If the engine already has ``rebuild_dp_group`` this is a no-op.
        """
        from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls

        engine = self.actor.engine

        if callable(getattr(engine, "rebuild_dp_group", None)):
            logger.debug("[ElasticActorWorker] Engine already has rebuild_dp_group, skipping patch")
            return

        strategy = self._get_actor_strategy()
        if strategy is None:
            logger.warning("[ElasticActorWorker] Cannot detect actor strategy; engine not patched")
            return

        original_cls = type(engine)
        try:
            elastic_cls = get_elastic_engine_cls(strategy, original_cls)
        except KeyError:
            logger.warning(f"[ElasticActorWorker] No elastic mixin for strategy={strategy!r}")
            return

        engine.__class__ = elastic_cls
        logger.info(f"[ElasticActorWorker] Engine patched: {original_cls.__name__} → {elastic_cls.__name__}")

    def _get_actor_strategy(self) -> Optional[str]:
        """Read actor strategy from config or infer from engine class name."""
        try:
            return self.config.actor.strategy
        except AttributeError:
            pass

        if self.actor and self.actor.engine:
            cls_name = type(self.actor.engine).__name__.lower()
            if "megatron" in cls_name:
                return "megatron"
            if "fsdp" in cls_name:
                return "fsdp2"

        return None

    # -------------------------------------------------------------------------
    # Training engine state transitions
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_train(self, new_train_world_ranks: list[int], param_version: int) -> bool:
        """
        Restore actor weights to GPU and rebuild DP group.

        Called **after** the rollout server on this resource has been put to
        sleep by ``ElasticAgentLoopManager`` (GPU memory already released).

        Steps:
        1. Load actor model weights from CPU back to GPU.
        2. Rebuild the DP communication group.

        Args:
            new_train_world_ranks: Global ranks forming the new training DP group.
            param_version: Parameter version to record in elastic state.

        Returns:
            True if successful, False otherwise.
        """
        assert self._elastic_state is not None, "call init_model() first"

        if self._elastic_state.current_mode == ElasticMode.TRAIN:
            logger.debug("[ElasticActorWorker] Already in TRAIN mode")
            return True

        logger.info(
            f"[ElasticActorWorker rank={dist.get_rank()}] "
            f"ROLLOUT → TRAIN  dp_size={len(new_train_world_ranks)}  param_version={param_version}"
        )

        try:
            # Step 1: Load actor model to GPU (rollout server already sleeping)
            self._load_actor_to_gpu()
            # Step 2: Rebuild DP group
            self._rebuild_dp_group(new_train_world_ranks)

            self._elastic_state.train_world_ranks = new_train_world_ranks
            self._elastic_state.param_version = param_version
            self._elastic_state.record_switch(ElasticMode.TRAIN)
            return True

        except Exception:
            logger.exception("[ElasticActorWorker] switch_to_train failed")
            self._elastic_state.is_healthy = False
            return False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_rollout(self, param_version: int) -> bool:
        """
        Offload actor weights to CPU so the rollout server can use the GPU.

        Called **before** the rollout server on this resource is woken up by
        ``ElasticAgentLoopManager``.

        Step:
        1. Offload actor model weights to CPU.

        Args:
            param_version: Parameter version to record in elastic state.

        Returns:
            True if successful, False otherwise.
        """
        assert self._elastic_state is not None, "call init_model() first"

        if self._elastic_state.current_mode == ElasticMode.ROLLOUT:
            logger.debug("[ElasticActorWorker] Already in ROLLOUT mode")
            return True

        logger.info(f"[ElasticActorWorker rank={dist.get_rank()}] TRAIN → ROLLOUT  param_version={param_version}")

        try:
            # Offload actor model so rollout server can reclaim GPU memory
            self._offload_actor_to_cpu()

            self._elastic_state.param_version = param_version
            self._elastic_state.record_switch(ElasticMode.ROLLOUT)
            return True

        except Exception:
            logger.exception("[ElasticActorWorker] switch_to_rollout failed")
            self._elastic_state.is_healthy = False
            return False

    # -------------------------------------------------------------------------
    # Weight synchronization
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def sync_weights_to_rollout_naive(self, global_steps: int = None):
        """Naive in-process weight sync for sleeping hybrid replicas (TRAIN mode).

        Called exclusively by ElasticCheckpointManager Path B for sleeping
        hybrid replicas.  Must NOT be confused with update_weights(), which is
        the NCCL trainer-side send called by Path A.

        The rollout server is sleeping (weights + kv_cache released).  This
        method temporarily restores weight buffers, writes the latest actor
        params, then releases them again — without waking the rollout server.
        """
        await self._update_weights_naive(global_steps=global_steps)

    async def _update_weights_naive(self, global_steps: int = None):
        """In-process weight sync for sleeping hybrid replicas (TRAIN mode).

        HYBRID sleep releases both weights and kv_cache from GPU.  Before we
        can copy the latest actor parameters into the rollout engine's weight
        buffers we must restore those buffers first.

        Flow
        ----
        1. resume(tags=["weights"])   – restore weight buffers to GPU
        2. get_per_tensor_param()     – pull latest actor params (on GPU)
        3. rollout.update_weights()   – copy actor params → rollout engine registry
        4. release_weights()          – free weight buffers (GPU memory returned to
                                        training engine); the rollout engine's internal
                                        weight registry retains the latest version so
                                        wake_up can restore them correctly.

        The rollout server stays asleep throughout (no kv_cache, no serving).
        We do NOT offload actor weights or resume kv_cache here — those are
        handled by the switch sequences (switch_to_rollout / wake_up) managed
        by ElasticAgentLoopManager.
        """
        from verl.utils.device import set_expandable_segments
        from verl.utils.profiler import log_gpu_memory_usage

        set_expandable_segments(False)
        log_gpu_memory_usage("Before naive update_weights (sleeping)", logger=logger)

        # 1. Restore rollout weight buffers to GPU (HYBRID sleep released them).
        await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights (sleeping)", logger=logger)

        # 2. & 3. Pull actor params and write into rollout weight buffers.
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )
        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = (not self.base_sync_done) or (
                self.rollout.sleep_level != 1 and self.config.rollout.free_cache_engine
            )

        if do_lora_base_sync:
            per_tensor_base_params, _ = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(per_tensor_base_params, peft_config=peft_config, base_sync_done=False)

        # 4. Release weight buffers to reclaim GPU memory for the training engine.
        #    The latest params have been written into the rollout engine's internal
        #    weight store (SGLang/vLLM maintain a registered-weights registry that
        #    survives release/resume cycles).
        #    On Train→Rollout switch, wake_up resumes weights from that registry
        #    (which now holds the latest version) and then resumes kv_cache.
        await self.rollout.release_weights()
        log_gpu_memory_usage("After release_weights (sleeping)", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)
        log_gpu_memory_usage("After naive update_weights (sleeping)", logger=logger)

    # -------------------------------------------------------------------------
    # DP group rebuild (pass-through to engine)
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the training DP process group.

        All ranks in the current global process group must call this
        simultaneously because ``dist.new_group`` is a collective.

        Args:
            new_world_ranks: Ordered list of global ranks in the new DP group.
        """
        """Delegate DP group rebuild to the (patched) engine."""
        if not callable(getattr(self.actor.engine, "rebuild_dp_group", None)):
            raise AttributeError(
                f"Engine {type(self.actor.engine).__name__} does not implement "
                "rebuild_dp_group(). Ensure the engine was patched via "
                "_patch_engine_to_elastic()."
            )
        self.actor.engine.rebuild_dp_group(new_world_ranks)

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
            "last_switch_time": s.last_switch_time,
            "total_switches": s.total_switches,
            "is_healthy": s.is_healthy,
        }

    # -------------------------------------------------------------------------
    # Private helpers – training engine only
    # -------------------------------------------------------------------------
    def _offload_actor_to_cpu(self) -> None:
        """
        Move actor model weights to CPU, freeing GPU memory for the rollout
        server.  Optimizer state is left on GPU (it will not be used during
        rollout mode) unless the engine's ``to()`` helper handles it.
        """
        if not (hasattr(self, "actor") and self.actor is not None):
            return
        if callable(getattr(self.actor.engine, "to", None)):
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        else:
            # Fallback: move module parameters manually
            engine = self.actor.engine
            if hasattr(engine, "module") and engine.module is not None:
                engine.module.cpu()
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        logger.info(f"[ElasticActorWorker rank={dist.get_rank()}] Actor model offloaded to CPU")

    def _load_actor_to_gpu(self) -> None:
        """
        Restore actor model weights from CPU back to GPU.

        The rollout server on this resource must already be sleeping before
        this is called so that GPU memory is available.
        """
        if not (hasattr(self, "actor") and self.actor is not None):
            return
        if callable(getattr(self.actor.engine, "to", None)):
            self.actor.engine.to("device", model=True, optimizer=False, grad=False)
        else:
            engine = self.actor.engine
            if hasattr(engine, "module") and engine.module is not None:
                device = torch.cuda.current_device()
                engine.module.to(device)
        logger.info(f"[ElasticActorWorker rank={dist.get_rank()}] Actor model loaded back to GPU")
