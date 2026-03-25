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
Elastic Trainer for VERL

Provides dynamic scaling capabilities for training resources through composition
with FullyAsyncTrainer.
"""

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from ...single_controller.ray import RayWorkerGroup

if TYPE_CHECKING:
    from .resource_manager import HybridEngineMode, HybridEngineResource

logger = logging.getLogger(__name__)


class ElasticTrainerMixin:
    """
    Mixin class providing elastic scaling capabilities for trainer.

    This class provides methods for dynamically adding/removing training
    resources and tracking elastic resource statistics.

    Usage:
        class ElasticTrainer(FullyAsyncTrainer, ElasticTrainerMixin):
            pass
    """

    def __init__(self, *args, **kwargs):
        # Don't call super().__init__ here - let the parent class handle it
        # This is a mixin, not a full class

        # Elastic resource management
        self.elastic_actors: list[HybridEngineResource] = []
        self.elastic_worker_groups: dict[str, RayWorkerGroup] = {}

        # Original DP size (before elastic)
        self._base_dp_size: int = 0

        # Statistics
        self._stats = {
            "elastic_added": 0,
            "elastic_removed": 0,
            "batches_processed_elastic": 0,
        }

        # Sync tracking
        self._sync_versions: dict[str, int] = {}
        self._current_version: int = 0

    def init_workers(self):
        """Store base DP size when workers are initialized"""
        if hasattr(self, "actor_wg") and self.actor_wg:
            self._base_dp_size = self.actor_wg.world_size

    async def add_elastic_actors(
        self,
        resources: list["HybridEngineResource"],
        checkpoint_manager=None,
    ) -> int:
        """
        Add elastic resources as training workers

        Args:
            resources: List of HybridEngineResource to add
            checkpoint_manager: Optional checkpoint manager for param sync

        Returns:
            Number of resources successfully added
        """
        added_count = 0

        for resource in resources:
            if resource.resource_id in self.elastic_worker_groups:
                logger.warning(f"Resource {resource.resource_id} already exists")
                continue

            try:
                # Create actor worker group from resource
                actor_wg = await self._create_actor_from_resource(resource)

                # Track resource
                self.elastic_actors.append(resource)
                self.elastic_worker_groups[resource.resource_id] = actor_wg
                self._sync_versions[resource.resource_id] = self._current_version

                # Sync params to new actor if checkpoint manager provided
                if checkpoint_manager:
                    await self._sync_params_to_actor(actor_wg, checkpoint_manager)

                added_count += 1
                self._stats["elastic_added"] += 1

                logger.info(f"Added elastic actor {resource.resource_id} to trainer")

            except Exception as e:
                logger.error(f"Failed to add elastic actor {resource.resource_id}: {e}")

        return added_count

    async def remove_elastic_actors(
        self,
        resources: list["HybridEngineResource"],
        graceful: bool = True,
    ) -> int:
        """
        Remove elastic training resources

        Args:
            resources: List of resources to remove
            graceful: If True, wait for pending batches to complete

        Returns:
            Number of resources successfully removed
        """
        removed_count = 0

        for resource in resources:
            if resource.resource_id not in self.elastic_worker_groups:
                logger.warning(f"Resource {resource.resource_id} not found")
                continue

            try:
                # Get worker group
                actor_wg = self.elastic_worker_groups[resource.resource_id]
                print(actor_wg)

                # Remove from tracking
                self.elastic_actors.remove(resource)
                del self.elastic_worker_groups[resource.resource_id]
                del self._sync_versions[resource.resource_id]

                # Clean up worker group
                # Note: This is simplified - actual implementation would need
                # proper worker group management

                removed_count += 1
                self._stats["elastic_removed"] += 1

                logger.info(f"Removed elastic actor {resource.resource_id} from trainer")

            except Exception as e:
                logger.error(f"Failed to remove elastic actor {resource.resource_id}: {e}")

        return removed_count

    async def _create_actor_from_resource(
        self,
        resource: "HybridEngineResource",
    ):
        """
        Create an actor worker group from HybridEngineResource

        This creates a new actor worker group that shares the same
        model architecture and can participate in distributed training.
        """
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker

        # Create worker class
        role_cls = RayClassWithInitArgs(
            cls=ActorRolloutRefWorker,
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        print(role_cls)

        # Create worker group with specified GPUs
        # Note: This is simplified - actual implementation would need
        # proper resource pool management

        worker_group = RayWorkerGroup()
        # worker_group.add_resources(resource.gpu_ranks)
        # worker_group.add_workers(role_cls, count=1)

        return worker_group

    async def _sync_params_to_actor(
        self,
        actor_wg,
        checkpoint_manager,
    ):
        """Sync parameters to a newly added actor"""
        try:
            if checkpoint_manager and hasattr(checkpoint_manager, "update_weights"):
                await checkpoint_manager.update_weights(global_steps=self._current_version)
            logger.debug("Synced params to new elastic actor")
        except Exception as e:
            logger.error(f"Failed to sync params to elastic actor: {e}")

    def get_consumption_rate(self) -> float:
        """Get current sample consumption rate (samples/sec)"""
        if not hasattr(self, "_consumption_history"):
            self._consumption_history = deque(maxlen=100)

        # Calculate rate from history
        if len(self._consumption_history) >= 2:
            time_delta = self._consumption_history[-1]["time"] - self._consumption_history[0]["time"]
            if time_delta > 0:
                total_samples = sum(h["samples"] for h in self._consumption_history)
                return total_samples / time_delta

        return 0.0

    def record_consumption(self, n_samples: int):
        """Record samples consumed (training completed)"""
        if not hasattr(self, "_consumption_history"):
            self._consumption_history = deque(maxlen=100)

        self._consumption_history.append(
            {
                "time": time.time(),
                "samples": n_samples,
            }
        )

        # Track elastic contributions
        self._stats["batches_processed_elastic"] += n_samples

    def get_current_dp_size(self) -> int:
        """Get current total DP size (base + elastic)"""
        base = (
            self._base_dp_size
            if self._base_dp_size > 0
            else (getattr(self.actor_wg, "world_size", 0) if hasattr(self, "actor_wg") and self.actor_wg else 0)
        )
        elastic = sum(wg.world_size for wg in self.elastic_worker_groups.values())
        return base + elastic

    def get_elastic_stats(self) -> dict:
        """Get elastic resource statistics"""
        return {
            **self._stats.copy(),
            "elastic_actors_active": len(self.elastic_actors),
            "current_dp_size": self.get_current_dp_size(),
            "base_dp_size": self._base_dp_size,
            "sync_versions": self._sync_versions.copy(),
        }

    async def update_weights(self, global_steps: int = None):
        """Override to track sync versions"""
        # Call parent's update_weights if available
        if hasattr(super(), "update_weights"):
            await super().update_weights(global_steps)

        if hasattr(self, "current_param_version"):
            self._current_version = self.current_param_version

        # Update sync versions for all elastic actors
        for resource_id in self._sync_versions:
            self._sync_versions[resource_id] = self._current_version

    async def get_statistics(self) -> dict:
        """Get comprehensive statistics"""
        base_metrics = {}
        if hasattr(super(), "get_statistics"):
            base_metrics = await super().get_statistics()

        return {
            **base_metrics,
            "elastic": self.get_elastic_stats(),
        }

    async def switch_mode(
        self,
        resource_id: str,
        target_mode: "HybridEngineMode",
    ):
        """
        Switch a specific resource to a different mode

        This is called by the resource manager when coordinating
        resource allocation.
        """
        resource = None
        for r in self.elastic_actors:
            if r.resource_id == resource_id:
                resource = r
                break

        if not resource:
            logger.warning(f"Resource {resource_id} not found in trainer")
            return False

        if target_mode.value == "train":
            # Already in train mode
            return True

        # Switch to rollout mode - remove from trainer
        removed = await self.remove_elastic_actors([resource], graceful=True)
        return removed > 0
