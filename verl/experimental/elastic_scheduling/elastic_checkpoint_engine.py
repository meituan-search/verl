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
Elastic Parameter Synchronization for VERL

Extends CheckpointEngineManager with hybrid replica support.

  self.replicas  (inherited from CheckpointEngineManager)
      = standalone replicas  +  awake hybrid replicas (ROLLOUT mode)
      Both groups use the base-class NCCL sync flow with release_kv_cache /
      resume_kv_cache so that model weights are never disturbed:

        abort_all_requests
        → release_kv_cache   (free kv-cache only, weights stay)
        → build NCCL group
        → trainer.update_weights() + replica.update_weights()
        → finalize
        → resume_kv_cache    (restore kv-cache)
        → resume_generation

  _sleep_hybrid_replicas  (sleeping hybrid replicas, TRAIN mode)
      GPU is held by the training engine; rollout server is sleeping with
      both weights and kv-cache released.  Use the naive (in-process) path:

        actor_wg.update_weights()

      The actor and rollout engine share the same Ray worker process.
      With checkpoint_engine.backend == "naive", this directly copies actor
      parameters into the rollout engine's weight buffers in GPU memory.
      No NCCL group is required and the rollout server stays asleep.

  Replica lifecycle
  -----------------
  add_hybrid_replicas(replicas, actor_wgs)
      Registers replicas in SLEEPING state by default:
        • stored in _sleep_hybrid_replicas (NOT added to self.replicas)
        • actor_wg stored for naive sync

  mark_hybrid_awake(resource_ids)   [Train → Rollout switch]
      • removes from _sleep_hybrid_replicas
      • calls super().add_replicas() → added to self.replicas (NCCL path)

  mark_hybrid_sleeping(resource_ids)  [Rollout → Train switch]
      • calls super().remove_replicas() → removed from self.replicas
      • adds to _sleep_hybrid_replicas (naive path)

  remove_hybrid_replicas(resource_ids)
      • removes from whichever set currently holds the replica

  update_weights(global_steps)
      1. super().update_weights()  – handles self.replicas (standalone + awake hybrid)
      2. naive sync loop           – handles _sleep_hybrid_replicas
"""

import logging

import ray

from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.config import CheckpointEngineConfig
from verl.workers.rollout import RolloutReplica

logger = logging.getLogger(__name__)


class ElasticCheckpointManager(CheckpointEngineManager):
    """
    Checkpoint manager for elastic scheduling.

    Replica sets managed by this class
    -----------------------------------
    self.replicas  (inherited)
        Standalone replicas  +  Awake hybrid replicas (ROLLOUT mode).
        The base-class update_weights() syncs all of them via NCCL using
        release_kv_cache / resume_kv_cache.

    self._sleep_hybrid_replicas
        Sleeping hybrid replicas (TRAIN mode).
        Synced via naive in-process actor_wg.update_weights(); no NCCL group.

    State transitions
    -----------------
    add_hybrid_replicas()   → initial state is SLEEPING (_sleep_hybrid_replicas)
    mark_hybrid_awake()     → SLEEPING → AWAKE  (move into self.replicas)
    mark_hybrid_sleeping()  → AWAKE → SLEEPING  (move out of self.replicas)
    remove_hybrid_replicas()→ removed from whichever set currently holds it
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        trainer: RayWorkerGroup,
        replicas: list[RolloutReplica],
    ) -> None:
        super().__init__(config=config, trainer=trainer, replicas=replicas)

        # resource_id → RolloutReplica  (sleeping hybrid replicas only)
        self._sleep_hybrid_replicas: dict[str, RolloutReplica] = {}
        # resource_id → RolloutReplica  (awake hybrid replicas only, also in self.replicas)
        self._awake_hybrid_replicas: dict[str, RolloutReplica] = {}
        # resource_id → RayWorkerGroup  (ElasticActorWorker wg, for naive sync)
        self._hybrid_actor_wgs: dict[str, RayWorkerGroup] = {}

    # -------------------------------------------------------------------------
    # Hybrid replica registry
    # -------------------------------------------------------------------------

    def add_hybrid_replicas(
        self,
        replicas: dict[str, RolloutReplica],
        actor_wgs: dict[str, RayWorkerGroup] | None = None,
    ) -> None:
        """
        Register hybrid replicas.  New replicas start in SLEEPING state.

        Sleeping replicas are NOT added to self.replicas; they will be moved
        there by mark_hybrid_awake() when the Train→Rollout switch happens.

        Args:
            replicas: Mapping of resource_id → RolloutReplica.
            actor_wgs: Optional mapping of resource_id → ElasticActorWorker
                RayWorkerGroup for naive in-process sync (TRAIN mode).
        """
        actor_wgs = actor_wgs or {}
        for resource_id, replica in replicas.items():
            self._sleep_hybrid_replicas[resource_id] = replica
            wg = actor_wgs.get(resource_id)
            if wg is not None:
                self._hybrid_actor_wgs[resource_id] = wg
        logger.info(
            f"[ElasticCheckpointManager] Registered {len(replicas)} hybrid replica(s) (SLEEPING): , ".join(
                replicas.keys()
            )
        )

    def remove_hybrid_replicas(self, resource_ids: list[str]) -> None:
        """
        Deregister hybrid replicas (from whichever state they are currently in).

        If a replica is AWAKE it is also removed from self.replicas via the
        base-class remove_replicas() so it is no longer synced over NCCL.

        Args:
            resource_ids: List of resource_ids to deregister.
        """
        awake_to_remove = [
            self._awake_hybrid_replicas[rid] for rid in resource_ids if rid in self._awake_hybrid_replicas
        ]
        if awake_to_remove:
            super().remove_replicas(awake_to_remove)

        for resource_id in resource_ids:
            self._awake_hybrid_replicas.pop(resource_id, None)
            self._sleep_hybrid_replicas.pop(resource_id, None)
            self._hybrid_actor_wgs.pop(resource_id, None)

        logger.info(
            f"[ElasticCheckpointManager] Deregistered {len(resource_ids)} hybrid replica(s): , ".join(resource_ids)
        )

    def mark_hybrid_awake(self, resource_ids: list[str] | str) -> None:
        """
        Transition hybrid replicas from SLEEPING → AWAKE (Train → Rollout switch).

        Replicas are moved from _sleep_hybrid_replicas into self.replicas via
        super().add_replicas() so the base-class NCCL sync picks them up on
        the next update_weights() call.

        Args:
            resource_ids: A single resource_id string or a list of them.
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]

        to_add: list[RolloutReplica] = []
        for rid in resource_ids:
            replica = self._sleep_hybrid_replicas.pop(rid, None)
            if replica is None:
                logger.warning(f"[ElasticCheckpointManager] mark_hybrid_awake: '{rid}' not in sleep set, skipping")
                continue
            self._awake_hybrid_replicas[rid] = replica
            to_add.append(replica)

        if to_add:
            super().add_replicas(to_add)

        logger.debug(f"[ElasticCheckpointManager] Marked as AWAKE: {resource_ids}")

    def mark_hybrid_sleeping(self, resource_ids: list[str] | str) -> None:
        """
        Transition hybrid replicas from AWAKE → SLEEPING (Rollout → Train switch).

        Replicas are removed from self.replicas via super().remove_replicas() and
        moved into _sleep_hybrid_replicas so the next update_weights() uses the
        naive in-process path.

        Args:
            resource_ids: A single resource_id string or a list of them.
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]

        to_remove: list[RolloutReplica] = []
        for rid in resource_ids:
            replica = self._awake_hybrid_replicas.pop(rid, None)
            if replica is None:
                logger.warning(f"[ElasticCheckpointManager] mark_hybrid_sleeping: '{rid}' not in awake set, skipping")
                continue
            self._sleep_hybrid_replicas[rid] = replica
            to_remove.append(replica)

        if to_remove:
            super().remove_replicas(to_remove)

        logger.debug(f"[ElasticCheckpointManager] Marked as SLEEPING: {resource_ids}")

    # -------------------------------------------------------------------------
    # Weight synchronization
    # -------------------------------------------------------------------------

    async def update_weights(self, global_steps: int = None) -> None:
        """
        Synchronize parameters from trainer to ALL replicas.

        Step 1 – NCCL path (delegated to base class)
            Handles self.replicas = standalone replicas + awake hybrid replicas.
            The base class calls release_kv_cache_replicas() before sync and
            resume_kv_cache_replicas() after, so model weights are never touched.

        Step 2 – Naive path (in-process)
            Handles _sleep_hybrid_replicas (TRAIN mode).
            Each sleeping replica is synced via its actor_wg.update_weights().
            No NCCL group required; processed serially.

        Args:
            global_steps: Current training step used as the parameter version tag.
        """
        nccl_replicas = len(self.replicas)
        sleep_replicas = len(self._sleep_hybrid_replicas)
        logger.info(
            f"[ElasticCheckpointManager] update_weights step={global_steps}: "
            f"NCCL replicas={nccl_replicas} (standalone+awake hybrid), "
            f"naive replicas={sleep_replicas} (sleeping hybrid)"
        )

        # Step 1: NCCL sync for standalone + awake hybrid replicas (base class)
        if self.replicas:
            await super().update_weights(global_steps)

        # Step 2: naive sync for sleeping hybrid replicas
        for resource_id, _replica in self._sleep_hybrid_replicas.items():
            actor_wg = self._hybrid_actor_wgs.get(resource_id)
            if actor_wg is None:
                logger.warning(
                    f"[ElasticCheckpointManager] No actor_wg for sleeping hybrid replica "
                    f"'{resource_id}'. Skipping sync."
                )
                continue
            try:
                ray.get(actor_wg.update_weights(global_steps=global_steps))
                logger.info(f"[ElasticCheckpointManager] Sleeping replica '{resource_id}' synced (naive, TRAIN mode).")
            except Exception as e:
                logger.exception(f"[ElasticCheckpointManager] Failed to sync sleeping replica '{resource_id}': {e}")
