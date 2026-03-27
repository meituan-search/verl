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
Elastic Agent Loop Manager for VERL

Extends FullyAsyncAgentLoopManager with dynamic server management:
- Add new rollout servers (when elastic resources switch to rollout mode)
- Remove rollout servers (when elastic resources switch to training mode)
- Update GlobalRequestLoadBalancer when server pool changes
- Coordinate parameter synchronization for newly added servers

All elastic replica lifecycle management (tracking, versioning, statistics)
is centralized here; ElasticRollouter only delegates to this class.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import ray
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.agent_loop import (
    FullyAsyncAgentLoopManager,
    FullyAsyncAgentLoopWorker,
)
from verl.single_controller.ray import RayWorkerGroup

if TYPE_CHECKING:
    from verl.workers.rollout.replica import RolloutReplica

logger = logging.getLogger(__name__)


@ray.remote
class ElasticGlobalRequestLoadBalancer:
    """
    Extended GlobalRequestLoadBalancer that supports dynamic server addition/removal.

    Adds:
    - add_server(): add a new server to the pool
    - remove_server(): mark a server for removal (no new requests)
    - cleanup_removed_server(): fully remove a drained server
    - get_inflight_count(): get the number of in-flight requests for a server
    """

    def __init__(self, server_actor_ids: list[str], max_cache_size: int = 10000):
        from cachetools import LRUCache

        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")

        self._inflight_requests: dict[str, int] = {sid: 0 for sid in server_actor_ids}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._removed_servers: set[str] = set()  # Servers being drained

    def acquire_server(self, request_id: str) -> str:
        """Acquire a server for the given request (sticky + least-loaded)."""
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            if server_id not in self._removed_servers:
                self._inflight_requests[server_id] += 1
                return server_id

        available = {sid: cnt for sid, cnt in self._inflight_requests.items() if sid not in self._removed_servers}
        if not available:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(available, key=available.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_server(self, server_id: str) -> None:
        """Add a new server to the load balancer pool."""
        if server_id in self._inflight_requests:
            self._removed_servers.discard(server_id)
            return
        self._inflight_requests[server_id] = 0
        self._removed_servers.discard(server_id)
        logger.info(f"[ElasticLB] Added server: {server_id}")

    def remove_server(self, server_id: str) -> None:
        """Mark server for removal (no new requests routed to it)."""
        if server_id in self._inflight_requests:
            self._removed_servers.add(server_id)
        logger.info(f"[ElasticLB] Marked server for removal: {server_id}")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def cleanup_removed_server(self, server_id: str) -> None:
        """Fully remove a server that has been drained."""
        self._inflight_requests.pop(server_id, None)
        self._removed_servers.discard(server_id)
        logger.info(f"[ElasticLB] Cleaned up server: {server_id}")

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return [sid for sid in self._inflight_requests if sid not in self._removed_servers]


class ElasticAgentLoopWorker(FullyAsyncAgentLoopWorker):
    def add_server(self, server_address: str, server_handle: ray.actor.ActorHandle) -> None:
        """
        Dynamically add a new rollout server to this worker's server manager.

        Called by ElasticAgentLoopManager when an elastic resource switches to rollout mode.
        After this call, the worker's load balancer can route requests to the new server.

        Args:
            server_address: The address/id of the new server (used as LB key).
            server_handle: The Ray actor handle for the new vLLM/SGLang server.
        """
        self.server_manager._server_id_to_handle[server_address] = server_handle
        logger.debug(f"[FullyAsyncAgentLoopWorker] Added server: {server_address}")

    def remove_server(self, server_address: str) -> None:
        """
        Remove a rollout server from this worker's server manager.

        Called by ElasticAgentLoopManager BEFORE abort_all_requests(), so that
        when FullyAsyncLLMServerManager resumes after stop_reason="aborted",
        the dead server handle is no longer in the map and the LB routes the
        resume request to a healthy server (partial rollout auto-resume).

        Args:
            server_address: The address/id of the server to remove.
        """
        self.server_manager._server_id_to_handle.pop(server_address, None)
        logger.debug(f"[FullyAsyncAgentLoopWorker] Removed server: {server_address}")


class ElasticAgentLoopManager(FullyAsyncAgentLoopManager):
    """
    Elastic Agent Loop Manager with dynamic server management.

    Extends FullyAsyncAgentLoopManager to support:
    1. Adding elastic rollout replicas when resources switch to rollout mode
    2. Removing elastic rollout replicas when resources switch to training mode
    3. Centralised tracking of replica lifecycle, param versions, and statistics

    All state for elastic replicas lives here; ElasticRollouter is a thin
    delegation layer that calls add_elastic_replica / remove_elastic_replica.

    Ordering guarantee for partial-rollout auto-resume on removal:
      LB mark-removing → workers remove handle → abort_all_requests → LB cleanup
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool=None,
        teacher_model_manager=None,
        reward_loop_worker_handles: list = None,
    ):
        # Use ElasticAgentLoopWorker so the worker class is clearly named
        # and can be extended with elastic-specific logic in the future.
        self.agent_loop_workers_class = ray.remote(ElasticAgentLoopWorker)
        super().__init__(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            teacher_model_manager=teacher_model_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        # resource_id -> RolloutReplica
        self._elastic_replicas: dict[str, RolloutReplica] = {}
        # resource_id -> server_address (for LB / worker notification)
        self._elastic_addresses: dict[str, str] = {}
        # resource_id -> param_version
        self._elastic_versions: dict[str, int] = {}

        # Timing / counters
        self._total_elastic_adds: int = 0
        self._total_elastic_removes: int = 0
        self._last_elastic_add_time: float = 0.0
        self._last_elastic_remove_time: float = 0.0

    # -------------------------------------------------------------------------
    # Override: use ElasticGlobalRequestLoadBalancer
    # -------------------------------------------------------------------------

    async def _init_global_load_balancer(self) -> None:
        """Override to use ElasticGlobalRequestLoadBalancer (supports add/remove)."""
        self.global_load_balancer = ElasticGlobalRequestLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
            max_cache_size=10000,
        )
        logger.info(
            f"[ElasticAgentLoopManager] Elastic load balancer initialised with {len(self.server_addresses)} servers"
        )

    # -------------------------------------------------------------------------
    # Public API – called by ElasticRollouter (and ElasticCoordinator)
    # -------------------------------------------------------------------------

    async def add_elastic_replica(
        self,
        resource_id: str,
        replica: "RolloutReplica",
        param_version: int,
    ) -> bool:
        """
        Add an elastic rollout replica to the production pool.

        Initialises the replica if needed, registers it with the load balancer,
        notifies all AgentLoopWorkers, and records tracking state.

        Args:
            resource_id: Unique identifier for the elastic resource.
            replica: RolloutReplica already switched to rollout mode.
            param_version: Parameter version the replica was synced to.

        Returns:
            True on success, False on failure.
        """
        if resource_id in self._elastic_replicas:
            logger.warning(f"[ElasticAgentLoopManager] Replica {resource_id} already registered, skipping")
            return False

        logger.info(f"[ElasticAgentLoopManager] Adding elastic replica {resource_id} (param_version={param_version})")
        try:
            # Ensure replica is initialised
            server_address = getattr(replica, "_server_address", None)
            server_handle = getattr(replica, "_server_handle", None)
            if server_address is None or server_handle is None:
                logger.info(f"[ElasticAgentLoopManager] Initialising replica {resource_id}...")
                await replica.init_standalone()
                server_address = replica._server_address
                server_handle = replica._server_handle

            # 1. Register with load balancer
            await self.global_load_balancer.add_server.remote(server_id=server_address)

            # 2. Notify AgentLoopWorkers
            await self._notify_workers_server_added(server_address, server_handle)

            # 3. Record state
            self._elastic_replicas[resource_id] = replica
            self._elastic_addresses[resource_id] = server_address
            self._elastic_versions[resource_id] = param_version
            self._total_elastic_adds += 1
            self._last_elastic_add_time = time.time()

            logger.info(
                f"[ElasticAgentLoopManager] Replica {resource_id} added at {server_address}. "
                f"Total elastic: {len(self._elastic_replicas)}"
            )
            return True

        except Exception as e:
            logger.error(f"[ElasticAgentLoopManager] Failed to add replica {resource_id}: {e}")
            return False

    async def remove_elastic_replica(self, resource_id: str) -> bool:
        """
        Remove an elastic rollout replica from the production pool.

        Ordering is critical for partial-rollout auto-resume:
          1. LB: mark server as removing (no NEW requests)
          2. Workers: remove server handle BEFORE abort
          3. abort_all_requests() → FullyAsyncLLMServerManager catches
             stop_reason="aborted" and re-submits with already-generated
             tokens → auto-resume on a healthy server
          4. LB: full cleanup

        Args:
            resource_id: Unique identifier of the elastic resource to remove.

        Returns:
            True on success, False if resource_id not found.
        """
        if resource_id not in self._elastic_replicas:
            logger.warning(f"[ElasticAgentLoopManager] Replica {resource_id} not found, skipping")
            return False

        server_address = self._elastic_addresses[resource_id]
        replica = self._elastic_replicas[resource_id]

        logger.info(f"[ElasticAgentLoopManager] Removing elastic replica {resource_id} at {server_address}")
        try:
            # Step 1: Stop new routing to this server
            await self.global_load_balancer.remove_server.remote(server_id=server_address)

            # Step 2: Remove handle from workers BEFORE aborting
            await self._notify_workers_server_removed(server_address)

            # Step 3: Abort in-flight requests → triggers partial-rollout resume
            try:
                await replica.abort_all_requests()
                logger.info(
                    f"[ElasticAgentLoopManager] Aborted in-flight requests on {server_address}; "
                    f"partial-rollout resume will redirect them to healthy servers"
                )
            except Exception as e:
                logger.warning(f"[ElasticAgentLoopManager] Error aborting requests on {server_address}: {e}")

            # Step 4: Full LB cleanup
            await self.global_load_balancer.cleanup_removed_server.remote(server_id=server_address)

            # Remove tracking state
            self._elastic_replicas.pop(resource_id)
            self._elastic_addresses.pop(resource_id)
            self._elastic_versions.pop(resource_id)
            self._total_elastic_removes += 1
            self._last_elastic_remove_time = time.time()

            logger.info(
                f"[ElasticAgentLoopManager] Replica {resource_id} removed. Total elastic: {len(self._elastic_replicas)}"
            )
            return True

        except Exception as e:
            logger.error(f"[ElasticAgentLoopManager] Failed to remove replica {resource_id}: {e}")
            return False

    def update_elastic_replica_version(self, resource_id: str, param_version: int) -> None:
        """Update the tracked parameter version for an elastic replica."""
        if resource_id in self._elastic_versions:
            self._elastic_versions[resource_id] = param_version
            logger.debug(f"[ElasticAgentLoopManager] Updated param version for {resource_id} to {param_version}")

    # -------------------------------------------------------------------------
    # Statistics / introspection
    # -------------------------------------------------------------------------

    def get_num_elastic_replicas(self) -> int:
        """Return the number of currently active elastic replicas."""
        return len(self._elastic_replicas)

    def get_elastic_replicas_info(self) -> list[dict]:
        """Return metadata for all active elastic replicas."""
        return [
            {
                "resource_id": rid,
                "server_address": self._elastic_addresses.get(rid, "unknown"),
                "param_version": self._elastic_versions.get(rid, -1),
            }
            for rid in self._elastic_replicas
        ]

    def get_elastic_statistics(self) -> dict:
        """Return elastic-specific counters for monitoring."""
        return {
            "elastic/num_elastic_replicas": len(self._elastic_replicas),
            "elastic/total_adds": self._total_elastic_adds,
            "elastic/total_removes": self._total_elastic_removes,
            "elastic/last_add_time": self._last_elastic_add_time,
            "elastic/last_remove_time": self._last_elastic_remove_time,
            "elastic/replica_param_versions": dict(self._elastic_versions),
        }

    def get_active_server_count(self) -> int:
        """Total active rollout servers (fixed + elastic)."""
        fixed = len(self.rollout_replicas) if hasattr(self, "rollout_replicas") else 0
        return fixed + len(self._elastic_replicas)

    def get_server_info(self) -> list[dict]:
        """Metadata for all active rollout servers."""
        servers = []
        if hasattr(self, "rollout_replicas"):
            for replica in self.rollout_replicas:
                servers.append(
                    {
                        "address": getattr(replica, "_server_address", "unknown"),
                        "type": "fixed",
                        "is_elastic": False,
                    }
                )
        for rid, address in self._elastic_addresses.items():
            servers.append(
                {
                    "address": address,
                    "resource_id": rid,
                    "type": "elastic",
                    "is_elastic": True,
                }
            )
        return servers

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _notify_workers_server_added(self, server_address: str, server_handle) -> None:
        """
        Notify all AgentLoopWorkers that a new server is available, and keep
        the manager-level server lists (server_addresses / server_handles) in sync.

        The manager-level lists are the source of truth used when spawning new
        workers later (e.g. after a scale-out event).  Each existing worker's
        FullyAsyncLLMServerManager._server_id_to_handle is updated via add_server().
        """
        # 1. Update manager-level server lists so future workers get the full pool.
        if server_address not in self.server_addresses:
            self.server_addresses.append(server_address)
            self.server_handles.append(server_handle)

        # 2. Notify each existing worker to add the new handle into its server_manager.
        if not getattr(self, "agent_loop_workers", None):
            return
        futures = []
        for worker in self.agent_loop_workers:
            try:
                futures.append(worker.add_server.remote(server_address, server_handle))
            except AttributeError:
                logger.debug("[ElasticAgentLoopManager] Worker doesn't support add_server()")
        if futures:
            results = await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in futures],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"[ElasticAgentLoopManager] Worker add_server failed: {result}")

    async def _notify_workers_server_removed(self, server_address: str) -> None:
        """
        Notify all AgentLoopWorkers that a server is no longer available, and
        remove it from the manager-level server lists.

        Must be called BEFORE abort_all_requests() so that when
        FullyAsyncLLMServerManager retries after stop_reason="aborted", the dead
        handle is already gone from every worker's _server_id_to_handle.
        """
        # 1. Update manager-level server lists.
        if server_address in self.server_addresses:
            idx = self.server_addresses.index(server_address)
            self.server_addresses.pop(idx)
            self.server_handles.pop(idx)

        # 2. Notify each existing worker to remove the stale handle.
        if not getattr(self, "agent_loop_workers", None):
            return
        futures = []
        for worker in self.agent_loop_workers:
            try:
                futures.append(worker.remove_server.remote(server_address))
            except AttributeError:
                logger.debug("[ElasticAgentLoopManager] Worker doesn't support remove_server()")
        if futures:
            await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in futures],
                return_exceptions=True,
            )

    async def _drain_server(self, server_address: str, timeout: float = 30.0) -> None:
        """Wait for in-flight requests to drain from a server (optional helper)."""
        start = time.time()
        last_log = start
        while time.time() - start < timeout:
            try:
                inflight = ray.get(self.global_load_balancer.get_inflight_count.remote(server_id=server_address))
                if inflight == 0:
                    logger.info(
                        f"[ElasticAgentLoopManager] Server {server_address} drained in {time.time() - start:.1f}s"
                    )
                    return
                if time.time() - last_log > 5.0:
                    logger.info(
                        f"[ElasticAgentLoopManager] Waiting for {inflight} in-flight requests on {server_address}..."
                    )
                    last_log = time.time()
            except Exception as e:
                logger.warning(f"[ElasticAgentLoopManager] Error checking inflight: {e}")
                break
            await asyncio.sleep(0.5)
        logger.warning(f"[ElasticAgentLoopManager] Drain timeout for {server_address} after {timeout}s")
