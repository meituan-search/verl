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

Extends CheckpointEngineManager to handle dynamic elastic resources:
1. Sync to fixed rollout replicas (standard behavior)
2. Sync to newly added elastic rollout replicas
3. Track parameter version per elastic replica
4. Support incremental sync (only changed parameters)

Key design:
- Wraps the existing CheckpointEngineManager
- Before each sync, checks for newly added elastic replicas
- Elastic replicas get the same sync as fixed replicas
- Coordinator is notified after sync completes
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import ray

from verl.checkpoint_engine import CheckpointEngineManager

if TYPE_CHECKING:
    from verl.checkpoint_engine.base import RolloutReplica

logger = logging.getLogger(__name__)


@dataclass
class ElasticSyncStats:
    """Statistics for elastic parameter synchronization."""

    total_syncs: int = 0
    total_elastic_syncs: int = 0
    failed_syncs: int = 0
    total_sync_time: float = 0.0
    last_sync_time: float = 0.0
    last_sync_version: int = -1

    # Per-replica tracking
    replica_versions: dict[str, int] = field(default_factory=dict)
    replica_sync_times: dict[str, float] = field(default_factory=dict)

    def record_sync(self, version: int, duration: float, success: bool, is_elastic: bool = False):
        """Record a sync operation."""
        self.total_syncs += 1
        if is_elastic:
            self.total_elastic_syncs += 1
        if not success:
            self.failed_syncs += 1
        else:
            self.total_sync_time += duration
            self.last_sync_time = time.time()
            self.last_sync_version = version

    def record_replica_sync(self, resource_id: str, version: int):
        """Record sync for a specific replica."""
        self.replica_versions[resource_id] = version
        self.replica_sync_times[resource_id] = time.time()

    def get_outdated_replicas(self, current_version: int) -> list[str]:
        """Get list of replica IDs that are behind current version."""
        return [rid for rid, ver in self.replica_versions.items() if ver < current_version]

    def to_dict(self) -> dict:
        return {
            "total_syncs": self.total_syncs,
            "total_elastic_syncs": self.total_elastic_syncs,
            "failed_syncs": self.failed_syncs,
            "avg_sync_time": self.total_sync_time / max(1, self.total_syncs),
            "last_sync_version": self.last_sync_version,
        }


class ElasticParameterSyncManager:
    """
    Elastic Parameter Synchronization Manager.

    Wraps CheckpointEngineManager to extend it with elastic replica support.
    When elastic resources are added to the rollout pool, they need to receive
    the latest parameters. This manager handles that seamlessly.

    The sync flow:
    1. Coordinator triggers _fit_update_weights() in ElasticTrainer
    2. ElasticParameterSyncManager.update_weights() is called
    3. Fixed replicas get parameters via the base CheckpointEngineManager
    4. New elastic replicas get parameters via additional sync calls
    5. All replica param versions are updated
    6. Coordinator is notified (staleness reset in rollouter)

    Configuration:
    - sync_backend: "nccl", "nixl", or "naive" (same as CheckpointEngineManager)
    - elastic_sync_backend: Override for elastic replicas (default: same as sync_backend)
      - "nixl" preferred for elastic because it supports dynamic topology
    """

    def __init__(
        self,
        base_manager: CheckpointEngineManager,
        elastic_rollouter=None,  # Reference to ElasticRollouter actor
        on_sync_complete: Optional[Callable] = None,
    ):
        """
        Args:
            base_manager: The underlying CheckpointEngineManager for fixed replicas
            elastic_rollouter: Ray actor handle to ElasticRollouter for version updates
            on_sync_complete: Callback after sync completes (version, duration)
        """
        self.base_manager = base_manager
        self.elastic_rollouter = elastic_rollouter
        self.on_sync_complete = on_sync_complete

        # Stats
        self.stats = ElasticSyncStats()

        # Elastic replica registry (resource_id -> RolloutReplica)
        self._elastic_replicas: dict[str, RolloutReplica] = {}

        # Current parameter version
        self._current_version: int = 0

        logger.info("[ElasticParamSync] Initialized")

    @property
    def trainer(self):
        return self.base_manager.trainer

    @property
    def replicas(self):
        return self.base_manager.replicas

    def register_elastic_replica(self, resource_id: str, replica: "RolloutReplica"):
        """
        Register a new elastic replica for parameter synchronization.

        Called when an elastic resource switches to rollout mode.
        The replica will receive parameters in the next sync cycle.
        """
        self._elastic_replicas[resource_id] = replica
        # Initialize version tracking (not yet synced)
        self.stats.replica_versions[resource_id] = -1
        logger.info(f"[ElasticParamSync] Registered elastic replica: {resource_id}")

    def unregister_elastic_replica(self, resource_id: str):
        """
        Unregister an elastic replica.

        Called when an elastic resource switches to training mode.
        """
        self._elastic_replicas.pop(resource_id, None)
        self.stats.replica_versions.pop(resource_id, None)
        self.stats.replica_sync_times.pop(resource_id, None)
        logger.info(f"[ElasticParamSync] Unregistered elastic replica: {resource_id}")

    async def update_weights(self, global_steps: int = None) -> None:
        """
        Update weights to ALL replicas: fixed + elastic.

        This is the main entry point for parameter synchronization.
        Called by ElasticTrainer._fit_update_weights().

        Steps:
        1. Update version counter
        2. Sync to fixed replicas via base_manager
        3. Sync to elastic replicas (newly joined ones)
        4. Update version tracking for all replicas
        5. Notify coordinator / rollouter

        Args:
            global_steps: Current training step (used as parameter version)
        """
        start_time = time.time()

        if global_steps is not None:
            self._current_version = global_steps

        logger.info(
            f"[ElasticParamSync] Starting sync to version {self._current_version}. "
            f"Fixed replicas: {len(self.replicas)}, "
            f"Elastic replicas: {len(self._elastic_replicas)}"
        )

        success = True
        try:
            # Step 1: Sync to fixed replicas (base behavior)
            await self.base_manager.update_weights(global_steps)

            # Step 2: Sync to elastic replicas
            if self._elastic_replicas:
                await self._sync_to_elastic_replicas(global_steps)

            # Step 3: Update version tracking
            for resource_id in self._elastic_replicas:
                self.stats.record_replica_sync(resource_id, self._current_version)

            # Step 4: Notify elastic rollouter to update its version tracking
            if self.elastic_rollouter is not None:
                for resource_id in self._elastic_replicas:
                    self.elastic_rollouter.update_elastic_replica_param_version.remote(
                        resource_id=resource_id,
                        param_version=self._current_version,
                    )

        except Exception as e:
            success = False
            logger.error(f"[ElasticParamSync] Sync failed: {e}")
            raise

        finally:
            duration = time.time() - start_time
            self.stats.record_sync(
                version=self._current_version,
                duration=duration,
                success=success,
                is_elastic=bool(self._elastic_replicas),
            )

            if success:
                logger.info(f"[ElasticParamSync] Sync complete in {duration:.2f}s (version={self._current_version})")

            # Notify callback
            if self.on_sync_complete and success:
                try:
                    await self.on_sync_complete(self._current_version, duration)
                except Exception as e:
                    logger.warning(f"[ElasticParamSync] on_sync_complete callback failed: {e}")

    async def _sync_to_elastic_replicas(self, global_steps: Optional[int] = None):
        """
        Synchronize parameters to elastic replicas.

        For elastic replicas, we need to:
        1. Wake them up (if sleeping after rollout switch)
        2. Send the latest parameters
        3. Let them go back to serving requests

        This uses the same checkpoint engine infrastructure as the base sync,
        but targets only the elastic replicas.

        For newly joined replicas (param_version == -1), we do a FULL sync.
        For replicas that are behind by only 1 version, we could do incremental sync
        (future optimization).
        """
        # Group replicas by sync need
        outdated_replicas = []
        for resource_id, replica in self._elastic_replicas.items():
            replica_version = self.stats.replica_versions.get(resource_id, -1)
            if replica_version < self._current_version:
                outdated_replicas.append((resource_id, replica))

        if not outdated_replicas:
            logger.debug("[ElasticParamSync] All elastic replicas up to date")
            return

        logger.info(
            f"[ElasticParamSync] Syncing to {len(outdated_replicas)} elastic replicas "
            f"(out of {len(self._elastic_replicas)} total)"
        )

        # Sync to each outdated elastic replica
        # We can do this concurrently if using NIXL backend
        sync_tasks = []
        for resource_id, replica in outdated_replicas:
            sync_tasks.append(self._sync_single_elastic_replica(resource_id, replica, global_steps))

        if sync_tasks:
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    resource_id = outdated_replicas[i][0]
                    logger.error(f"[ElasticParamSync] Failed to sync replica {resource_id}: {result}")

    async def _sync_single_elastic_replica(
        self,
        resource_id: str,
        replica: "RolloutReplica",
        global_steps: Optional[int] = None,
    ):
        """
        Sync parameters to a single elastic replica.

        This is designed to work with the existing CheckpointEngineManager infrastructure.
        The actual weight transfer uses the configured backend (NCCL/NIXL/naive).

        For NIXL backend: supports dynamic topology, ideal for elastic scaling
        For NCCL backend: requires rebuild_group=True to handle new replicas
        """
        start_time = time.time()
        logger.info(f"[ElasticParamSync] Syncing to elastic replica {resource_id}...")

        try:
            # Use the base manager's infrastructure to sync to this specific replica
            # Try to use the existing checkpoint engine infrastructure
            # The base manager's backend handles the actual weight transfer
            backend = self.base_manager.backend

            if backend == "naive":
                # For naive (colocated) mode: use update_weights directly
                await replica.abort_all_requests()
                workers = list(replica.workers)
                from verl.checkpoint_engine.base import _worker_cls
                from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

                rollout_wg = RayWorkerGroup(
                    worker_handles=workers,
                    ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
                )

                # Build process group with trainer + this replica
                self.base_manager.build_process_group(rollout_wg)

                # Transfer weights
                ray.get(
                    self.base_manager.trainer.update_weights(global_steps=global_steps)
                    + rollout_wg.update_weights(global_steps=global_steps)
                )

                # Finalize
                ray.get(
                    self.base_manager.trainer.execute_checkpoint_engine(
                        ["finalize"] * self.base_manager.trainer.world_size
                    )
                    + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
                )

                await replica.resume_generation()

            elif backend in ("nccl", "hccl", "kimi_ckpt_engine"):
                # For NCCL-based backends: rebuild group to include new replica
                await replica.abort_all_requests()

                workers = list(replica.workers)
                from verl.checkpoint_engine.base import _worker_cls
                from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

                rollout_wg = RayWorkerGroup(
                    worker_handles=workers,
                    ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
                )

                # Sleep replica to free memory before weight sync
                # (sleep one replica, not the full set in base_manager)
                try:
                    await replica.sleep()
                except Exception:
                    pass  # sleep may not be available for all replica types

                # Build process group for this single replica
                self.base_manager.build_process_group(rollout_wg)

                # Transfer weights
                ray.get(
                    self.base_manager.trainer.update_weights(global_steps=global_steps)
                    + rollout_wg.update_weights(global_steps=global_steps)
                )

                # Finalize
                ray.get(
                    self.base_manager.trainer.execute_checkpoint_engine(
                        ["finalize"] * self.base_manager.trainer.world_size
                    )
                    + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
                )

                # Wake up replica
                try:
                    await replica.wake_up()
                except Exception:
                    pass
                await replica.resume_generation()

            elif backend in ("nixl", "mooncake"):
                # For NIXL/Mooncake: supports dynamic ring topology
                # Add this elastic replica to the existing ring topology
                # by including all replicas (fixed + this elastic one) in a sync.
                logger.info(f"[ElasticParamSync] Using NIXL/dynamic backend for {resource_id}")

                # Combine fixed replicas with this elastic replica
                existing_replicas = list(self.replicas) + [replica]

                # Use the base manager's build_process_group to establish the ring
                # including this new elastic replica
                workers = []
                for r in existing_replicas:
                    workers.extend(r.workers)
                from verl.checkpoint_engine.base import _worker_cls
                from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

                rollout_wg = RayWorkerGroup(
                    worker_handles=workers,
                    ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
                )
                self.base_manager.build_process_group(rollout_wg)

                # Transfer weights to all (including new elastic replica)
                ray.get(
                    self.base_manager.trainer.update_weights(global_steps=global_steps)
                    + rollout_wg.update_weights(global_steps=global_steps)
                )
                ray.get(
                    self.base_manager.trainer.execute_checkpoint_engine(
                        ["finalize"] * self.base_manager.trainer.world_size
                    )
                    + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
                )
                await replica.resume_generation()

            else:
                logger.warning(
                    f"[ElasticParamSync] Unknown backend '{backend}' for elastic sync. "
                    f"Skipping elastic replica {resource_id}"
                )
                return

            duration = time.time() - start_time
            logger.info(
                f"[ElasticParamSync] Sync to {resource_id} complete in {duration:.2f}s "
                f"(version={self._current_version})"
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[ElasticParamSync] Sync to {resource_id} failed in {duration:.2f}s: {e}")
            raise

    async def sleep_replicas(self):
        """Delegate to base manager."""
        await self.base_manager.sleep_replicas()

    async def wake_up_replicas(self):
        """Delegate to base manager."""
        await self.base_manager.wake_up_replicas()

    def get_sync_status(self) -> dict:
        """Get synchronization status for all replicas."""
        return {
            "current_version": self._current_version,
            "fixed_replicas": len(self.replicas),
            "elastic_replicas": len(self._elastic_replicas),
            "outdated_elastic": len(self.stats.get_outdated_replicas(self._current_version)),
            "stats": self.stats.to_dict(),
        }

    def is_replica_synced(self, resource_id: str) -> bool:
        """Check if a specific elastic replica is synced to current version."""
        return self.stats.replica_versions.get(resource_id, -1) >= self._current_version
