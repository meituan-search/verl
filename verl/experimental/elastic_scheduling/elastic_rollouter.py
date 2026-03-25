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
Elastic Rollouter for VERL

Provides dynamic scaling capabilities for rollout resources through composition
with FullyAsyncRollouter.
"""

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from verl.workers.rollout.replica import RolloutReplica

if TYPE_CHECKING:
    from .resource_manager import HybridEngineMode, HybridEngineResource

logger = logging.getLogger(__name__)


class ElasticRollouterMixin:
    """
    Mixin class providing elastic scaling capabilities for rollout.

    This class provides methods for dynamically adding/removing rollout
    resources and tracking elastic resource statistics.

    Usage:
        class ElasticRollouter(FullyAsyncRollouter, ElasticRollouterMixin):
            pass
    """

    def __init__(self, *args, **kwargs):
        # Don't call super().__init__ here - let the parent class handle it
        # This is a mixin, not a full class

        # Elastic resource management
        self.elastic_resources: list[HybridEngineResource] = []
        self.elastic_replicas: dict[str, RolloutReplica] = {}

        # Pending samples tracking
        self._pending_samples: deque = deque()
        self._processing_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "elastic_added": 0,
            "elastic_removed": 0,
            "samples_from_elastic": 0,
        }

    async def add_elastic_resources(
        self,
        resources: list["HybridEngineResource"],
        checkpoint_manager=None,
    ) -> int:
        """
        Add elastic resources as rollout workers

        Args:
            resources: List of HybridEngineResource to add
            checkpoint_manager: Optional checkpoint manager for param sync

        Returns:
            Number of resources successfully added
        """
        added_count = 0

        for resource in resources:
            if resource.resource_id in self.elastic_replicas:
                logger.warning(f"Resource {resource.resource_id} already exists")
                continue

            try:
                # Create rollout replica from resource
                replica = await self._create_replica_from_resource(resource)

                # Register with async rollout manager
                if hasattr(self, "async_rollout_manager"):
                    self.async_rollout_manager.add_replicas([replica])

                # Track resource
                self.elastic_resources.append(resource)
                self.elastic_replicas[resource.resource_id] = replica

                # Sync params to new replica if checkpoint manager provided
                if checkpoint_manager:
                    await self._sync_params_to_replica(replica, checkpoint_manager)

                added_count += 1
                self._stats["elastic_added"] += 1

                logger.info(f"Added elastic resource {resource.resource_id} to rollout")

            except Exception as e:
                logger.error(f"Failed to add elastic resource {resource.resource_id}: {e}")

        # Update concurrent sample limit
        if added_count > 0 and hasattr(self, "max_concurrent_samples"):
            self.max_concurrent_samples = min(
                self.max_concurrent_samples + added_count * 16, getattr(self, "max_required_samples", 10000)
            )

        return added_count

    async def remove_elastic_resources(
        self,
        resources: list["HybridEngineResource"],
        graceful: bool = True,
    ) -> int:
        """
        Remove elastic rollout resources

        Args:
            resources: List of resources to remove
            graceful: If True, wait for pending samples to complete

        Returns:
            Number of resources successfully removed
        """
        removed_count = 0

        for resource in resources:
            if resource.resource_id not in self.elastic_replicas:
                logger.warning(f"Resource {resource.resource_id} not found")
                continue

            try:
                if graceful:
                    # Wait for pending samples
                    await self._wait_for_pending_samples(resource)

                # Get replica
                replica = self.elastic_replicas[resource.resource_id]

                # Remove from manager
                if hasattr(self, "async_rollout_manager"):
                    self.async_rollout_manager.remove_replicas([replica])

                # Clean up
                self.elastic_resources.remove(resource)
                del self.elastic_replicas[resource.resource_id]

                removed_count += 1
                self._stats["elastic_removed"] += 1

                logger.info(f"Removed elastic resource {resource.resource_id} from rollout")

            except Exception as e:
                logger.error(f"Failed to remove elastic resource {resource.resource_id}: {e}")

        # Update concurrent sample limit
        if removed_count > 0 and hasattr(self, "max_concurrent_samples"):
            self.max_concurrent_samples = max(
                self.max_concurrent_samples - removed_count * 16,
                len(getattr(self, "async_rollout_manager", {}).server_handles or []) * 16,
            )

        return removed_count

    async def _create_replica_from_resource(self, resource: "HybridEngineResource"):
        """
        Create a RolloutReplica from HybridEngineResource

        This creates a new rollout replica that shares the actor weights
        with the main rollout pool.
        """
        from verl.workers.rollout.replica import get_rollout_replica_class

        # Get rollout config
        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model

        # Create replica
        replica_rank = len(self.elastic_replicas)
        replica_class = get_rollout_replica_class(rollout_config.name, rollout_config.mode)

        replica = replica_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=len(resource.gpu_ranks),
        )

        # Initialize with existing worker group (hybrid mode)
        if hasattr(self, "rollout_wg") and self.rollout_wg:
            await replica.init_hybrid(self.rollout_wg)
        else:
            await replica.init_standalone()

        return replica

    async def _sync_params_to_replica(
        self,
        replica,
        checkpoint_manager,
    ):
        """Sync parameters to a newly added replica"""
        try:
            if checkpoint_manager and hasattr(checkpoint_manager, "update_weights"):
                await checkpoint_manager.update_weights(global_steps=getattr(self, "global_steps", 0))
            logger.debug("Synced params to replica")
        except Exception as e:
            logger.error(f"Failed to sync params to replica: {e}")

    async def _wait_for_pending_samples(
        self,
        resource: "HybridEngineResource",
        timeout: float = 30.0,
    ):
        """Wait for pending samples on a resource to complete"""
        replica = self.elastic_replicas.get(resource.resource_id)
        if not replica:
            return

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if replica is still processing
            if hasattr(replica, "is_processing") and not replica.is_processing:
                return
            await asyncio.sleep(0.1)

        logger.warning(f"Timeout waiting for pending samples on {resource.resource_id}")

    def get_production_rate(self) -> float:
        """Get current sample production rate (samples/sec)"""
        if not hasattr(self, "_production_history"):
            self._production_history = deque(maxlen=100)

        # Calculate rate from history
        if len(self._production_history) >= 2:
            time_delta = self._production_history[-1]["time"] - self._production_history[0]["time"]
            if time_delta > 0:
                total_samples = sum(h["samples"] for h in self._production_history)
                return total_samples / time_delta

        return 0.0

    def record_production(self, n_samples: int):
        """Record samples produced"""
        if not hasattr(self, "_production_history"):
            self._production_history = deque(maxlen=100)

        self._production_history.append(
            {
                "time": time.time(),
                "samples": n_samples,
            }
        )

        # Track elastic contributions
        self._stats["samples_from_elastic"] += n_samples

    def get_queue_stats(self) -> dict:
        """Get queue statistics"""
        queue_stats = {}
        if hasattr(self, "message_queue_client") and self.message_queue_client:
            queue_stats = self.message_queue_client.get_statistics_sync() or {}

        return {
            "queue_size": queue_stats.get("queue_size", 0),
            "queue_capacity": getattr(self, "max_queue_size", 0),
            "queue_utilization": queue_stats.get("queue_size", 0) / max(getattr(self, "max_queue_size", 1), 1),
            "pending_queue_size": getattr(self.pending_queue, "qsize", lambda: 0)(),
            "active_tasks": len(getattr(self, "active_tasks", [])),
        }

    def get_elastic_stats(self) -> dict:
        """Get elastic resource statistics"""
        return {
            **self._stats.copy(),
            "elastic_resources_active": len(self.elastic_resources),
            "elastic_replicas": len(self.elastic_replicas),
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
        for r in self.elastic_resources:
            if r.resource_id == resource_id:
                resource = r
                break

        if not resource:
            logger.warning(f"Resource {resource_id} not found in rollouter")
            return False

        if target_mode.value == "rollout":
            # Already in rollout mode
            return True

        # Switch to train mode - remove from rollouter
        removed = await self.remove_elastic_resources([resource], graceful=True)
        return removed > 0
