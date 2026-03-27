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

Extends FullyAsyncRollouter with dynamic rollout DP management:
- Supports adding/removing rollout DP instances (elastic resources)
- Integrates with ElasticAgentLoopManager for dynamic server management
- Provides hooks for the ElasticCoordinator to trigger elastic scaling
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType

if TYPE_CHECKING:
    from verl.workers.rollout.replica import RolloutReplica

logger = logging.getLogger(__name__)


class ElasticRollouter(FullyAsyncRollouter):
    """
    Elastic Rollouter with dynamic DP management.

    Extends FullyAsyncRollouter to support:
    1. Dynamic addition of elastic rollout replicas
    2. Dynamic removal of elastic rollout replicas
    3. Load balancer update when replicas change
    4. Production rate tracking for coordinator decisions

    The key difference from FullyAsyncRollouter:
    - Uses ElasticAgentLoopManager instead of FullyAsyncAgentLoopManager
    - Provides add_elastic_replica() and remove_elastic_replica() methods
    - Tracks per-replica production statistics

    Architecture:
        ElasticRollouter (Ray actor)
            ├── Fixed Rollout Replicas (always active)
            └── Elastic Rollout Replicas (dynamically added/removed)
                ↓
            ElasticAgentLoopManager
                ├── GlobalRequestLoadBalancer (updated dynamically)
                └── AgentLoopWorkers (updated dynamically)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=device_name,
        )

        # Elastic replica management
        self._elastic_replicas: dict[str, RolloutReplica] = {}  # resource_id -> replica
        self._elastic_replica_versions: dict[str, int] = {}  # resource_id -> param_version

        # Production statistics per replica
        self._replica_production_counts: dict[str, int] = {}

        # Scaling state
        self._scaling_lock = asyncio.Lock()
        self._is_scaling = False

        # Timing statistics
        self._last_elastic_add_time: float = 0.0
        self._last_elastic_remove_time: float = 0.0
        self._total_elastic_adds: int = 0
        self._total_elastic_removes: int = 0

        # Total production tracking (for ElasticCoordinator rate monitoring)
        self._total_produced_samples: int = 0

        logger.info("[ElasticRollouter] Initialized with elastic DP support")

    async def _init_async_rollout_manager(self):
        """
        Initialize elastic agent loop manager instead of standard one.

        Override to use ElasticAgentLoopManager which supports dynamic
        server addition and removal.
        """
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        assert self.config.actor_rollout_ref.rollout.mode == "async"

        from verl.experimental.elastic_scheduling.elastic_agent_loop import ElasticAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager: ElasticAgentLoopManager = await ElasticAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        logger.info(
            f"[ElasticRollouter] ElasticAgentLoopManager initialized with "
            f"{len(self.async_rollout_manager.rollout_replicas)} fixed replicas"
        )

    # =========================================================================
    # Elastic Replica Management (called by ElasticCoordinator)
    # =========================================================================

    async def add_elastic_replica(
        self,
        resource_id: str,
        replica: "RolloutReplica",
        param_version: int,
    ) -> bool:
        """
        Add an elastic rollout replica to the production pool.

        This is called by ElasticCoordinator when an elastic resource
        switches from Train to Rollout mode.

        Args:
            resource_id: Unique identifier for the elastic resource
            replica: The RolloutReplica to add (already initialized)
            param_version: The parameter version this replica was synced to

        Returns:
            True if successfully added, False otherwise
        """
        async with self._scaling_lock:
            if resource_id in self._elastic_replicas:
                logger.warning(f"[ElasticRollouter] Elastic replica {resource_id} already registered")
                return False

            logger.info(f"[ElasticRollouter] Adding elastic replica {resource_id} (param_version={param_version})")

            try:
                # Add to ElasticAgentLoopManager
                await self.async_rollout_manager.add_rollout_server(replica)

                # Track the replica
                self._elastic_replicas[resource_id] = replica
                self._elastic_replica_versions[resource_id] = param_version
                self._replica_production_counts[resource_id] = 0

                self._last_elastic_add_time = time.time()
                self._total_elastic_adds += 1

                # Update concurrent sample limit to account for new replica
                if self.max_concurrent_samples is not None:
                    # Allow ~16 concurrent samples per replica (same as initialization)
                    extra_slots = 16
                    async with self.lock:
                        self.max_concurrent_samples += extra_slots

                logger.info(
                    f"[ElasticRollouter] Successfully added elastic replica {resource_id}. "
                    f"Total elastic replicas: {len(self._elastic_replicas)}"
                )
                return True

            except Exception as e:
                logger.error(f"[ElasticRollouter] Failed to add elastic replica {resource_id}: {e}")
                return False

    async def remove_elastic_replica(self, resource_id: str, graceful: bool = True) -> bool:
        """
        Remove an elastic rollout replica from the production pool.

        This is called by ElasticCoordinator when an elastic resource
        switches from Rollout to Train mode.

        Args:
            resource_id: Unique identifier for the elastic resource to remove
            graceful: If True, wait for in-flight requests to complete

        Returns:
            True if successfully removed, False otherwise
        """
        async with self._scaling_lock:
            if resource_id not in self._elastic_replicas:
                logger.warning(f"[ElasticRollouter] Elastic replica {resource_id} not found")
                return False

            logger.info(f"[ElasticRollouter] Removing elastic replica {resource_id} (graceful={graceful})")

            try:
                replica = self._elastic_replicas[resource_id]

                # Remove from ElasticAgentLoopManager (stops new routing to this server)
                server_address = getattr(replica, "_server_address", None)
                if server_address:
                    await self.async_rollout_manager.remove_rollout_server(
                        server_address=server_address,
                        graceful=graceful,
                    )

                # Remove tracking
                del self._elastic_replicas[resource_id]
                del self._elastic_replica_versions[resource_id]
                del self._replica_production_counts[resource_id]

                self._last_elastic_remove_time = time.time()
                self._total_elastic_removes += 1

                # Update concurrent sample limit
                if self.max_concurrent_samples is not None:
                    async with self.lock:
                        self.max_concurrent_samples = max(
                            16,  # minimum concurrent samples
                            self.max_concurrent_samples - 16,
                        )

                logger.info(
                    f"[ElasticRollouter] Successfully removed elastic replica {resource_id}. "
                    f"Total elastic replicas: {len(self._elastic_replicas)}"
                )
                return True

            except Exception as e:
                logger.error(f"[ElasticRollouter] Failed to remove elastic replica {resource_id}: {e}")
                return False

    async def update_elastic_replica_param_version(self, resource_id: str, param_version: int):
        """
        Update the tracked parameter version for an elastic replica.

        Called after ElasticParamSync completes a sync to this replica.
        """
        async with self.lock:
            if resource_id in self._elastic_replica_versions:
                self._elastic_replica_versions[resource_id] = param_version
                logger.debug(f"[ElasticRollouter] Updated param version for {resource_id} to {param_version}")

    # =========================================================================
    # Statistics & Monitoring
    # =========================================================================

    async def get_elastic_statistics(self) -> dict:
        """Get elastic-specific statistics."""
        base_stats = await self.get_statistics()

        elastic_stats = {
            "elastic/num_elastic_replicas": len(self._elastic_replicas),
            "elastic/total_adds": self._total_elastic_adds,
            "elastic/total_removes": self._total_elastic_removes,
            "elastic/last_add_time": self._last_elastic_add_time,
            "elastic/last_remove_time": self._last_elastic_remove_time,
            "elastic/replica_param_versions": dict(self._elastic_replica_versions),
        }

        return {**base_stats, **elastic_stats}

    def get_num_active_replicas(self) -> int:
        """Get total number of active rollout replicas (fixed + elastic)."""
        fixed_replicas = (
            len(self.async_rollout_manager.rollout_replicas) if self.async_rollout_manager is not None else 0
        )
        return fixed_replicas + len(self._elastic_replicas)

    def get_elastic_replicas_info(self) -> list[dict]:
        """Get information about all elastic replicas."""
        return [
            {
                "resource_id": rid,
                "param_version": self._elastic_replica_versions.get(rid, -1),
                "production_count": self._replica_production_counts.get(rid, 0),
            }
            for rid in self._elastic_replicas
        ]

    async def reset_staleness(self):
        """
        Reset staleness after parameter update.

        Extended to also update elastic replica param versions.
        """
        # Reset base staleness
        timing_raw = await super().reset_staleness()

        # Update elastic replica versions to current trainer param version
        # (they should have been synced before this reset)
        async with self.lock:
            for resource_id in self._elastic_replica_versions:
                # The param versions will be updated by ElasticParamSync
                # We just log here for debugging
                logger.debug(f"[ElasticRollouter] Post-reset param versions: {self._elastic_replica_versions}")

        return timing_raw

    def record_produced_samples(self, n_samples: int):
        """
        Record that N samples were produced and put into the message queue.

        Called internally by the rollout loop whenever samples are enqueued.
        Used by ElasticCoordinator to monitor production rate.
        """
        self._total_produced_samples += n_samples

    def get_total_produced_samples(self) -> int:
        """
        Get total number of samples produced since start.

        Used by ElasticCoordinator to compute production rate.
        Returns the base class's total_generated_samples if available,
        which counts all samples actually generated and put into the queue.
        """
        # Use the base class's counter if available (more accurate)
        if hasattr(self, "total_generated_samples"):
            return self.total_generated_samples
        return self._total_produced_samples

    def get_rollout_server_address(self) -> str:
        """
        Get the server address for rollout serving.

        Used by ElasticCoordinator when creating rollout replicas from
        elastic worker handles.
        """
        if hasattr(self, "async_rollout_manager") and self.async_rollout_manager is not None:
            server_info = self.async_rollout_manager.get_server_info()
            if server_info:
                return server_info[0].get("address", "")
        return ""


# ============================================================================
# Elastic Agent Loop Manager
# ============================================================================


class ElasticAgentLoopManagerMixin:
    """
    Mixin to extend AgentLoopManager with dynamic server management.

    Provides add_rollout_server() and remove_rollout_server() methods
    that update the load balancer and agent loop workers.

    This is mixed into AgentLoopManager (or its subclasses).
    """

    async def add_rollout_server(self, replica: "RolloutReplica"):
        """
        Dynamically add a new rollout server to the pool.

        Steps:
        1. Initialize the replica (standalone mode for elastic workers)
        2. Get server handle and address
        3. Update GlobalRequestLoadBalancer with new server
        4. Notify all AgentLoopWorkers of the new server

        Args:
            replica: The RolloutReplica to add (already mode-switched to rollout)
        """
        server_address = getattr(replica, "_server_address", None)
        server_handle = getattr(replica, "_server_handle", None)

        if server_address is None or server_handle is None:
            # Initialize the replica if not already done
            try:
                await replica.init_standalone()
                server_address = replica._server_address
                server_handle = replica._server_handle
            except Exception as e:
                logger.error(f"[ElasticAgentLoopManagerMixin] Failed to init replica: {e}")
                raise

        logger.info(f"[ElasticAgentLoopManagerMixin] Adding server at {server_address}")

        # Update load balancer - add new server
        if hasattr(self, "global_load_balancer"):
            try:
                await self.global_load_balancer.add_server.remote(server_id=server_address)
            except Exception as e:
                logger.warning(f"[ElasticAgentLoopManagerMixin] Failed to update load balancer: {e}")

        # Update server list
        if not hasattr(self, "_server_id_to_handle"):
            self._server_id_to_handle = {}
        self._server_id_to_handle[server_address] = server_handle

        # Update AgentLoopWorkers
        if hasattr(self, "agent_loop_workers"):
            update_futures = []
            for worker in self.agent_loop_workers:
                if hasattr(worker, "add_server"):
                    update_futures.append(worker.add_server.remote(server_address, server_handle))
            if update_futures:
                await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in update_futures])

        # Track replica
        if not hasattr(self, "_elastic_server_replicas"):
            self._elastic_server_replicas = {}
        self._elastic_server_replicas[server_address] = replica

        logger.info(f"[ElasticAgentLoopManagerMixin] Server {server_address} added successfully")

    async def remove_rollout_server(self, server_address: str, graceful: bool = True):
        """
        Remove a rollout server from the pool.

        Ordering is critical for partial-rollout auto-resume to work correctly.
        When a server is removed during an active rollout, in-flight requests are
        aborted. FullyAsyncLLMServerManager catches stop_reason="aborted" and
        re-submits the request with already-generated tokens (partial rollout resume).
        For this resume to route to a *healthy* server, the dead server's handle must
        be removed from AgentLoopWorkers BEFORE the abort is issued.

        Correct ordering:
        1. LB: mark server as removing (no NEW requests routed here)
        2. Notify workers: remove server handle from _server_id_to_handle  ← BEFORE abort
        3. abort_all_requests() → triggers partial rollout resume on other servers
        4. LB: full cleanup

        Args:
            server_address: Address of the server to remove
            graceful: Ignored. We always abort immediately to trigger partial-rollout
                resume on other servers, avoiding a long drain wait during elastic switch.
        """
        logger.info(f"[ElasticAgentLoopManagerMixin] Removing server at {server_address}")

        # Step 1: Remove from load balancer (no new requests routed here)
        if hasattr(self, "global_load_balancer"):
            try:
                await self.global_load_balancer.remove_server.remote(server_id=server_address)
            except Exception as e:
                logger.warning(f"[ElasticAgentLoopManagerMixin] Failed to remove from load balancer: {e}")

        # Step 2: Notify AgentLoopWorkers to remove the handle BEFORE abort.
        # Resume requests from FullyAsyncLLMServerManager will then be routed only
        # to healthy servers since the dead handle is no longer in _server_id_to_handle.
        if hasattr(self, "agent_loop_workers"):
            remove_futures = []
            for worker in self.agent_loop_workers:
                if hasattr(worker, "remove_server"):
                    remove_futures.append(worker.remove_server.remote(server_address))
            if remove_futures:
                await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in remove_futures], return_exceptions=True)

        # Also remove from local handle map
        if hasattr(self, "_server_id_to_handle"):
            self._server_id_to_handle.pop(server_address, None)

        # Step 3: Abort all in-flight requests.
        # FullyAsyncLLMServerManager.generate() catches stop_reason="aborted" and
        # re-submits with already-generated prefix tokens → auto-resume on healthy server.
        if hasattr(self, "_elastic_server_replicas"):
            replica = self._elastic_server_replicas.pop(server_address, None)
            if replica:
                try:
                    await replica.abort_all_requests()
                    logger.info(
                        f"[ElasticAgentLoopManagerMixin] Aborted in-flight requests on {server_address}; "
                        f"partial-rollout resume redirects them to healthy servers"
                    )
                except Exception as e:
                    logger.warning(f"[ElasticAgentLoopManagerMixin] Error aborting requests: {e}")

        logger.info(f"[ElasticAgentLoopManagerMixin] Server {server_address} removed")

    async def _wait_server_drain(self, server_address: str, timeout: float = 30.0):
        """
        Wait for in-flight requests to a server to complete.

        Args:
            server_address: Server to drain
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check load balancer inflight count
                if hasattr(self, "global_load_balancer"):
                    inflight = await self.global_load_balancer.get_inflight_count.remote(server_id=server_address)
                    if inflight == 0:
                        break
            except Exception:
                break

            await asyncio.sleep(0.5)

        elapsed = time.time() - start_time
        logger.info(f"[ElasticAgentLoopManagerMixin] Server {server_address} drained in {elapsed:.1f}s")
