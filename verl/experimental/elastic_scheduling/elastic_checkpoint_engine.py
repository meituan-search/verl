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
        - Sleeping replicas are temporarily woken up, synced, then put back
          to sleep so the training engine can reclaim their GPUs.
        - Awake replicas follow the same abort → sync → resume flow as
          standalone replicas, but through a dedicated per-replica group.
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

    - add_hybrid_replica(resource_id, replica)   – register a hybrid replica
    - remove_hybrid_replica(resource_id)          – deregister a hybrid replica
    - update_weights(global_steps)               – sync ALL replicas in one call

    Hybrid replica sync (per replica):
      awake replica   → abort → sleep → build group → sync → wake_up → resume
      sleeping replica→        sleep* → build group → sync → sleep   (stay asleep)

    * A sleeping replica's sleep() is idempotent; calling it again is safe.

    The per-replica group approach (one NCCL group per hybrid replica) avoids
    the need for a global topology rebuild that would require all replicas to
    participate simultaneously.
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

    # -------------------------------------------------------------------------
    # Hybrid replica registry
    # -------------------------------------------------------------------------

    def add_hybrid_replica(self, resource_id: str, replica: RolloutReplica) -> None:
        """
        Register a hybrid replica for parameter synchronization.

        Call this when an elastic resource switches to rollout mode (or at
        init time for pre-bound replicas).  The replica will receive parameters
        on the next update_weights() call regardless of its current sleep state.

        Args:
            resource_id: Unique identifier (e.g. "elastic_0").
            replica: The RolloutReplica object (init_hybrid already called).
        """
        self._hybrid_replicas[resource_id] = replica
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
        logger.info(f"[ElasticCheckpointManager] Deregistered hybrid replica: {resource_id}")

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
            barrier across all replicas.  The flow per hybrid replica is:

              Awake replica (currently serving requests):
                1. abort_all_requests()   – save in-flight state
                2. sleep()                – release GPU memory for weight sync
                3. build_process_group()  – establish trainer ↔ replica channel
                4. trainer.update_weights() + replica.update_weights()
                5. finalize()             – tear down the temp channel
                6. wake_up()              – restore kv-cache
                7. resume_generation()    – replay saved in-flight requests

              Sleeping replica (GPU held by training engine):
                1. sleep()                – (idempotent, no-op if already asleep)
                2. build_process_group()
                3. trainer.update_weights() + replica.update_weights()
                4. finalize()
                5. sleep()                – ensure weights are released again

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

        Handles both the awake and sleeping cases transparently.  The caller
        (update_weights) is responsible for logging aggregate errors.

        Args:
            resource_id: Identifier used only for logging.
            replica: The hybrid RolloutReplica to synchronize.
            global_steps: Parameter version to tag the replica with.
        """
        logger.info(f"[ElasticCheckpointManager] Syncing hybrid replica '{resource_id}'...")

        is_awake = getattr(replica, "_is_awake", None)
        # If the replica doesn't track sleep state, assume we need to abort
        # in-flight requests (conservative / safe default).
        needs_abort_resume = is_awake is not False

        if self.backend == "naive":
            # Naive (colocated) backend: in-place weight update, no NCCL group needed.
            if needs_abort_resume:
                await replica.abort_all_requests()
            ray.get(self.trainer.update_weights(global_steps=global_steps))
            if needs_abort_resume:
                await replica.resume_generation()
            return

        # For all other backends (nccl, hccl, nixl, mooncake, …):
        # 1. Abort in-flight requests if the replica is awake.
        if needs_abort_resume:
            await replica.abort_all_requests()

        # 2. Sleep the replica to free GPU memory for the weight transfer.
        #    sleep() is idempotent – calling it on an already-sleeping replica is safe.
        await replica.sleep()

        # 3. Build a temporary trainer ↔ this-replica process group.
        rollout_wg = RayWorkerGroup(
            worker_handles=list(replica.workers),
            ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
        )
        self.build_process_group(rollout_wg)

        # 4. Transfer weights.
        ray.get(
            self.trainer.update_weights(global_steps=global_steps)
            + rollout_wg.update_weights(global_steps=global_steps)
        )

        # 5. Finalize (tear down temp NCCL group / release buffers).
        ray.get(
            self.trainer.execute_checkpoint_engine(["finalize"] * self.trainer.world_size)
            + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
        )

        # 6. Restore the replica to its pre-sync state.
        if needs_abort_resume:
            # Was awake: wake up and resume serving requests.
            await replica.wake_up()
            await replica.resume_generation()
        else:
            # Was sleeping: keep it asleep so the training engine retains the GPU.
            await replica.sleep()

        logger.info(f"[ElasticCheckpointManager] Hybrid replica '{resource_id}' synced.")
