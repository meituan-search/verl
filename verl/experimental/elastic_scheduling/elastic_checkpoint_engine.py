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

Extends CheckpointEngineManager with hybrid replica support:

  Standalone replicas  (self.replicas, inherited)
      Fixed rollout servers with dedicated GPUs.  Handled by the base
      CheckpointEngineManager.update_weights() unchanged.

  Hybrid replicas  (self._hybrid_replicas)
      Share GPUs with the training engine.  May be sleeping (GPU held by
      trainer) or awake (serving requests) at the time of a sync.
      All hybrid replicas — regardless of current sleep state — receive the
      latest parameters on every update_weights() call:

        - Awake replicas (ROLLOUT mode): actor weights are offloaded to CPU.
          Follow the same abort → sleep → NCCL sync → wake_up → resume flow
          as standalone replicas, through a dedicated per-replica NCCL group.

        - Sleeping replicas (TRAIN mode): GPU is held by the training engine.
          Use in-process sync via the corresponding ElasticActorWorker.update_weights().
          This is equivalent to "naive" backend: the actor and rollout engine share
          the same process, so weights are copied directly in GPU memory without
          any NCCL communication group.  The rollout server stays asleep throughout.

  The per-replica group approach (one NCCL group per awake hybrid replica) avoids
  the need for a global topology rebuild that would require all replicas to
  participate simultaneously.

  Key insight: because update_weights() is called periodically for ALL hybrid
  replicas regardless of sleep state, the rollout engine always holds the latest
  parameters when an elastic resource is woken up for serving.  No extra sync is
  needed after wake_up() during an elastic switch.
"""

import asyncio
import logging

import ray

from verl.checkpoint_engine import CheckpointEngineManager
from verl.checkpoint_engine.base import _worker_cls
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.workers.config import CheckpointEngineConfig
from verl.workers.rollout import RolloutReplica

logger = logging.getLogger(__name__)


class ElasticCheckpointManager(CheckpointEngineManager):
    """
    Checkpoint manager that handles both standalone and hybrid rollout replicas.

    Inherits the full standalone-replica sync pipeline from CheckpointEngineManager
    and adds:

    - add_hybrid_replica(resource_id, replica, actor_wg)  – register a hybrid replica
    - remove_hybrid_replica(resource_id)                  – deregister a hybrid replica
    - update_weights(global_steps)                        – sync ALL replicas in one call

    Hybrid replica sync (per replica):
      awake replica (ROLLOUT mode)
          → abort_all_requests → sleep → build NCCL group → sync → wake_up → resume
      sleeping replica (TRAIN mode)
          → actor_wg.update_weights()   (in-process, no NCCL needed)

    The per-replica group approach (one NCCL group per awake hybrid replica) avoids
    the need for a global topology rebuild that would require all replicas to
    participate simultaneously.

    For sleeping replicas, the corresponding ElasticActorWorker (actor_wg) is called
    directly.  Since the actor and rollout engine share the same Ray worker process,
    this is an in-GPU-memory copy with no inter-process communication overhead.
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        trainer: RayWorkerGroup,
        replicas: list[RolloutReplica],
    ) -> None:
        super().__init__(config=config, trainer=trainer, replicas=replicas)
        # resource_id -> RolloutReplica  (hybrid replicas only)
        self._hybrid_replicas: dict[str, RolloutReplica] = {}
        # resource_id -> RayWorkerGroup  (ElasticActorWorker wg for each hybrid replica)
        # Used for in-process weight sync when the replica is sleeping (TRAIN mode).
        self._hybrid_actor_wgs: dict[str, RayWorkerGroup] = {}
        # Set of resource_ids that are currently AWAKE (ROLLOUT mode).
        # Managed by mark_hybrid_awake() / mark_hybrid_sleeping() called from ElasticTrainer
        # during elastic switch sequences.  Replicas NOT in this set are treated as sleeping
        # (TRAIN mode) and synced via in-process actor_wg.update_weights().
        self._awake_hybrid_replicas: set[str] = set()

    # -------------------------------------------------------------------------
    # Hybrid replica registry
    # -------------------------------------------------------------------------

    def add_hybrid_replica(
        self,
        resource_id: str,
        replica: RolloutReplica,
        actor_wg: RayWorkerGroup | None = None,
    ) -> None:
        """
        Register a hybrid replica for parameter synchronization.

        Call this when an elastic resource switches to rollout mode (or at
        init time for pre-bound replicas).  The replica will receive parameters
        on the next update_weights() call regardless of its current sleep state.

        Args:
            resource_id: Unique identifier (e.g. "elastic_0").
            replica: The RolloutReplica object (init_hybrid already called).
            actor_wg: The ElasticActorWorker RayWorkerGroup that backs this
                hybrid replica.  Required for in-process weight sync when the
                replica is sleeping (TRAIN mode).  If None, sleeping-replica
                sync is skipped with a warning.
        """
        self._hybrid_replicas[resource_id] = replica
        if actor_wg is not None:
            self._hybrid_actor_wgs[resource_id] = actor_wg
        logger.info(f"[ElasticCheckpointManager] Registered hybrid replica: {resource_id}")

    def remove_hybrid_replica(self, resource_id: str) -> None:
        """
        Deregister a hybrid replica.

        Call this when an elastic resource switches back to training mode.
        Subsequent update_weights() calls will no longer include this replica.

        Args:
            resource_id: Unique identifier of the replica to remove.
        """
        self._hybrid_replicas.pop(resource_id, None)
        self._hybrid_actor_wgs.pop(resource_id, None)
        self._awake_hybrid_replicas.discard(resource_id)
        logger.info(f"[ElasticCheckpointManager] Deregistered hybrid replica: {resource_id}")

    def mark_hybrid_awake(self, resource_id: str) -> None:
        """
        Mark a hybrid replica as AWAKE (ROLLOUT mode).

        Call this after a Train→Rollout switch completes (i.e., after
        ElasticAgentLoopManager.add_elastic_replica() succeeds).
        On the next update_weights(), this replica will use the NCCL sync path.

        Args:
            resource_id: Unique identifier of the elastic resource.
        """
        self._awake_hybrid_replicas.add(resource_id)
        logger.debug(f"[ElasticCheckpointManager] Marked hybrid replica '{resource_id}' as AWAKE")

    def mark_hybrid_sleeping(self, resource_id: str) -> None:
        """
        Mark a hybrid replica as SLEEPING (TRAIN mode).

        Call this after a Rollout→Train switch completes (i.e., after
        ElasticAgentLoopManager.remove_elastic_replica() succeeds).
        On the next update_weights(), this replica will use the in-process
        actor_wg.update_weights() path instead of NCCL.

        Args:
            resource_id: Unique identifier of the elastic resource.
        """
        self._awake_hybrid_replicas.discard(resource_id)
        logger.debug(f"[ElasticCheckpointManager] Marked hybrid replica '{resource_id}' as SLEEPING")

    # -------------------------------------------------------------------------
    # Weight synchronization
    # -------------------------------------------------------------------------

    async def update_weights(self, global_steps: int = None) -> None:
        """
        Synchronize parameters from trainer to ALL replicas.

        Standalone replicas (self.replicas)
            Delegated to base CheckpointEngineManager.update_weights() unchanged.

        Hybrid replicas (self._hybrid_replicas)
            Each replica is synced independently to avoid a global collective
            barrier across all replicas.  The flow per hybrid replica depends on
            the replica's current mode:

              Awake replica (ROLLOUT mode – actor offloaded to CPU):
                1. abort_all_requests()   – save in-flight state
                2. sleep()                – release kv-cache GPU memory
                3. build_process_group()  – establish NCCL trainer ↔ replica channel
                4. trainer.update_weights() + replica.update_weights()
                5. finalize()             – tear down the temp NCCL channel
                6. wake_up()              – restore kv-cache
                7. resume_generation()    – replay saved in-flight requests

              Sleeping replica (TRAIN mode – GPU held by training engine):
                actor_wg.update_weights()  – in-process sync via ElasticActorWorker.
                The actor and rollout engine share the same Ray worker process, so
                weights are copied directly in GPU memory.  No NCCL group is needed
                and the rollout server remains asleep throughout.

        Args:
            global_steps: Current training step, used as the parameter version
                tag propagated to the rollout servers.
        """
        # Step 1: sync standalone replicas via base class
        if self.replicas:
            await super().update_weights(global_steps)

        # Step 2: sync all hybrid replicas (sleeping and awake)
        if self._hybrid_replicas:
            sync_tasks = [
                self._sync_hybrid_replica(resource_id, replica, global_steps)
                for resource_id, replica in self._hybrid_replicas.items()
            ]
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            for resource_id, result in zip(self._hybrid_replicas, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"[ElasticCheckpointManager] Failed to sync hybrid replica '{resource_id}': {result}")

    async def _sync_hybrid_replica(
        self,
        resource_id: str,
        replica: RolloutReplica,
        global_steps: int | None,
    ) -> None:
        """
        Synchronize parameters to a single hybrid replica.

        The sync path depends on the current mode of the elastic resource:

        ROLLOUT mode (replica is awake, actor weights offloaded to CPU):
            Follows the same NCCL-based flow as a standalone replica.
            A temporary NCCL process group is built between the trainer and the
            rollout workers of this replica; weights are pushed over NCCL; then
            the group is torn down and the replica resumes serving.

        TRAIN mode (replica is sleeping, GPU held by training engine):
            Uses in-process sync via the registered ElasticActorWorker wg.
            The actor and rollout engine share the same Ray worker process, so
            ``actor_wg.update_weights()`` copies weights directly in GPU memory
            without any inter-process communication.  The rollout server stays
            asleep throughout.

        Args:
            resource_id: Identifier used only for logging.
            replica: The hybrid RolloutReplica to synchronize.
            global_steps: Parameter version to tag the replica with.
        """
        logger.info(f"[ElasticCheckpointManager] Syncing hybrid replica '{resource_id}'...")

        # Determine sleep state from the authoritative _awake_hybrid_replicas set.
        # This set is maintained by mark_hybrid_awake() / mark_hybrid_sleeping() which
        # are called by ElasticTrainer during elastic switch sequences.
        # Default (not in set): treat as sleeping (TRAIN mode) — safe conservative choice
        # since an in-process sync on a sleeping replica is always safe.
        is_sleeping = resource_id not in self._awake_hybrid_replicas

        # -----------------------------------------------------------------------
        # Path A: TRAIN mode (sleeping) – in-process sync via ElasticActorWorker
        # -----------------------------------------------------------------------
        if is_sleeping:
            actor_wg = self._hybrid_actor_wgs.get(resource_id)
            if actor_wg is None:
                logger.warning(
                    f"[ElasticCheckpointManager] No actor_wg for sleeping hybrid replica "
                    f"'{resource_id}'. Skipping in-process sync (register via add_hybrid_replica)."
                )
                return

            # actor_wg.update_weights() calls ActorRolloutRefWorker.update_weights() which
            # runs in the ElasticActorWorker process.  When checkpoint_engine.backend==naive
            # (i.e. hybrid/colocated mode), this directly copies actor params → rollout engine
            # via the shared in-process rollout handle.  No NCCL group is required.
            ray.get(actor_wg.update_weights(global_steps=global_steps))
            logger.info(f"[ElasticCheckpointManager] Hybrid replica '{resource_id}' synced (in-process, TRAIN mode).")
            return

        # -----------------------------------------------------------------------
        # Path B: ROLLOUT mode (awake) – NCCL sync (same as standalone replicas)
        # -----------------------------------------------------------------------

        # 1. Abort in-flight requests to save partial rollout state.
        await replica.abort_all_requests()

        # 2. Sleep the replica to free kv-cache GPU memory for the weight transfer.
        await replica.sleep()

        # 3. Build a temporary trainer ↔ this-replica NCCL process group.
        rollout_wg = RayWorkerGroup(
            worker_handles=list(replica.workers),
            ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
        )
        self.build_process_group(rollout_wg)

        # 4. Transfer weights over NCCL.
        ray.get(
            self.trainer.update_weights(global_steps=global_steps)
            + rollout_wg.update_weights(global_steps=global_steps)
        )

        # 5. Finalize (tear down temp NCCL group / release buffers).
        ray.get(
            self.trainer.execute_checkpoint_engine(["finalize"] * self.trainer.world_size)
            + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
        )

        # 6. Restore: wake up and resume serving requests.
        await replica.wake_up()
        await replica.resume_generation()

        logger.info(f"[ElasticCheckpointManager] Hybrid replica '{resource_id}' synced (NCCL, ROLLOUT mode).")
