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

Extends AgentLoopManager with dynamic server management:
- Add new rollout servers (when elastic resources switch to rollout mode)
- Remove rollout servers (when elastic resources switch to training mode)
- Update GlobalRequestLoadBalancer when server pool changes
- Coordinate parameter synchronization for newly added servers
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import ray
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager
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
    - remove_server(): remove a server from the pool (no new requests)
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
        # Sticky session: reuse same server for multi-turn conversations
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            if server_id not in self._removed_servers:
                self._inflight_requests[server_id] += 1
                return server_id

        # Route to least loaded server (exclude servers being drained)
        available = {sid: count for sid, count in self._inflight_requests.items() if sid not in self._removed_servers}
        if not available:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(available, key=available.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return  # Server was removed, ignore
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_server(self, server_id: str) -> None:
        """Add a new server to the load balancer pool."""
        if server_id in self._inflight_requests:
            # Re-enable if it was being drained
            self._removed_servers.discard(server_id)
            return

        self._inflight_requests[server_id] = 0
        self._removed_servers.discard(server_id)
        logger.info(f"[ElasticLB] Added server: {server_id}")

    def remove_server(self, server_id: str) -> None:
        """
        Mark server for removal (no new requests routed to it).

        Existing in-flight requests continue until complete.
        Server is fully removed when get_inflight_count returns 0.
        """
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


class ElasticAgentLoopManager(FullyAsyncAgentLoopManager):
    """
    Elastic Agent Loop Manager with dynamic server management.

    Extends FullyAsyncAgentLoopManager to support:
    1. Adding new rollout servers when elastic resources switch to rollout mode
    2. Removing rollout servers when elastic resources switch to training mode
    3. Coordinating parameter sync for newly added servers

    The load balancer is extended to support add_server/remove_server operations.
    AgentLoopWorkers are updated via remote calls when the server pool changes.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool=None,
        teacher_model_manager=None,
        reward_loop_worker_handles: list = None,
    ):
        super().__init__(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            teacher_model_manager=teacher_model_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        # Elastic server tracking
        self._elastic_replicas: dict[str, RolloutReplica] = {}  # address -> replica

    async def _init_global_load_balancer(self) -> None:
        """
        Override to use ElasticGlobalRequestLoadBalancer.

        The elastic load balancer supports dynamic add/remove operations.
        """
        self.global_load_balancer = ElasticGlobalRequestLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
            max_cache_size=10000,
        )
        logger.info(
            f"[ElasticAgentLoopManager] Elastic load balancer initialized with {len(self.server_addresses)} servers"
        )

    async def add_rollout_server(self, replica: "RolloutReplica") -> bool:
        """
        Dynamically add a new rollout server to the pool.

        Steps:
        1. Initialize replica if not already initialized
        2. Register server with elastic load balancer
        3. Notify all AgentLoopWorkers of the new server
        4. Track the replica for future management

        Args:
            replica: The RolloutReplica to add (should be in rollout mode)

        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Get or initialize server handle and address
            server_address = getattr(replica, "_server_address", None)
            server_handle = getattr(replica, "_server_handle", None)

            if server_address is None or server_handle is None:
                logger.info("[ElasticAgentLoopManager] Initializing new elastic replica...")
                await replica.init_standalone()
                server_address = replica._server_address
                server_handle = replica._server_handle

            logger.info(f"[ElasticAgentLoopManager] Adding server at {server_address}")

            # Update load balancer
            await self.global_load_balancer.add_server.remote(server_id=server_address)

            # Update AgentLoopWorkers
            await self._notify_workers_server_added(server_address, server_handle)

            # Track replica
            self._elastic_replicas[server_address] = replica

            # Track in server handle map
            if not hasattr(self, "_server_id_to_handle"):
                self._server_id_to_handle = {}

            logger.info(f"[ElasticAgentLoopManager] Server {server_address} added successfully")
            return True

        except Exception as e:
            logger.error(f"[ElasticAgentLoopManager] Failed to add server: {e}")
            return False

    async def remove_rollout_server(
        self,
        server_address: str,
        graceful: bool = True,
        drain_timeout: float = 30.0,
    ) -> bool:
        """
        Remove a rollout server from the pool, with correct partial-rollout resume ordering.

        The ordering here is critical for partial rollout auto-resume to work correctly:

        1. Mark server in LB as "removing" → stops NEW requests being routed to it.
        2. Notify AgentLoopWorkers to remove server from their _server_id_to_handle.
           → This must happen BEFORE abort so that when FullyAsyncLLMServerManager
             retries after receiving stop_reason="aborted", the LB can only route
             the resume request to remaining healthy servers (old handle is gone).
        3. abort_all_requests() → triggers stop_reason="aborted" on all in-flight
             requests; FullyAsyncLLMServerManager catches this and automatically
             re-submits with already-generated tokens (partial rollout resume).
        4. Cleanup LB entry fully.

        Args:
            server_address: Address of server to remove
            graceful: Unused in elastic mode. We always abort immediately to unblock
                rollout workers via partial-rollout resume. Kept for API compatibility.
            drain_timeout: Unused (same reason). Kept for API compatibility.

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            logger.info(f"[ElasticAgentLoopManager] Removing server {server_address} (partial-rollout resume ordering)")

            # Step 1: Mark for removal in LB (no new requests routed here)
            await self.global_load_balancer.remove_server.remote(server_id=server_address)

            # Step 2: Notify workers to remove the server handle BEFORE aborting.
            # This ensures that when FullyAsyncLLMServerManager resumes after abort,
            # its _server_id_to_handle no longer contains the dead server, and the
            # LB will correctly route the resume to a healthy server.
            await self._notify_workers_server_removed(server_address)

            # Step 3: Abort all in-flight requests on this server.
            # FullyAsyncLLMServerManager.generate() catches stop_reason="aborted"
            # and automatically re-submits with the already-generated prefix tokens,
            # continuing generation on another server (partial rollout auto-resume).
            replica = self._elastic_replicas.get(server_address)
            if replica:
                try:
                    await replica.abort_all_requests()
                    logger.info(
                        f"[ElasticAgentLoopManager] Aborted in-flight requests on {server_address}; "
                        f"partial-rollout resume will redirect them to healthy servers"
                    )
                except Exception as e:
                    logger.warning(f"[ElasticAgentLoopManager] Error aborting requests: {e}")

            # Step 4: Fully clean up LB entry
            await self.global_load_balancer.cleanup_removed_server.remote(server_id=server_address)

            # Remove from tracking
            self._elastic_replicas.pop(server_address, None)

            logger.info(f"[ElasticAgentLoopManager] Server {server_address} removed successfully")
            return True

        except Exception as e:
            logger.error(f"[ElasticAgentLoopManager] Failed to remove server {server_address}: {e}")
            return False

    async def _notify_workers_server_added(self, server_address: str, server_handle):
        """Notify all AgentLoopWorkers that a new server was added."""
        if not hasattr(self, "agent_loop_workers") or not self.agent_loop_workers:
            return

        notify_futures = []
        for worker in self.agent_loop_workers:
            # Try to call add_server on the worker
            try:
                notify_futures.append(worker.add_server.remote(server_address, server_handle))
            except AttributeError:
                logger.debug("[ElasticAgentLoopManager] Worker doesn't support add_server()")

        if notify_futures:
            results = await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in notify_futures],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"[ElasticAgentLoopManager] Worker add_server failed: {result}")

    async def _notify_workers_server_removed(self, server_address: str):
        """Notify all AgentLoopWorkers that a server was removed."""
        if not hasattr(self, "agent_loop_workers") or not self.agent_loop_workers:
            return

        notify_futures = []
        for worker in self.agent_loop_workers:
            try:
                notify_futures.append(worker.remove_server.remote(server_address))
            except AttributeError:
                logger.debug("[ElasticAgentLoopManager] Worker doesn't support remove_server()")

        if notify_futures:
            await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in notify_futures],
                return_exceptions=True,
            )

    async def _drain_server(self, server_address: str, timeout: float = 30.0):
        """
        Wait for in-flight requests to drain from a server.

        Args:
            server_address: Server to drain
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()
        last_log = start_time

        while time.time() - start_time < timeout:
            try:
                inflight = ray.get(self.global_load_balancer.get_inflight_count.remote(server_id=server_address))
                if inflight == 0:
                    logger.info(
                        f"[ElasticAgentLoopManager] Server {server_address} drained in {time.time() - start_time:.1f}s"
                    )
                    return

                # Log progress periodically
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

    def get_active_server_count(self) -> int:
        """Get number of currently active rollout servers."""
        fixed_count = len(self.rollout_replicas) if hasattr(self, "rollout_replicas") else 0
        elastic_count = len(self._elastic_replicas)
        return fixed_count + elastic_count

    def get_server_info(self) -> list[dict]:
        """Get information about all active servers."""
        servers = []

        # Fixed servers
        if hasattr(self, "rollout_replicas"):
            for replica in self.rollout_replicas:
                servers.append(
                    {
                        "address": getattr(replica, "_server_address", "unknown"),
                        "type": "fixed",
                        "is_elastic": False,
                    }
                )

        # Elastic servers
        for address, replica in self._elastic_replicas.items():
            servers.append(
                {
                    "address": address,
                    "type": "elastic",
                    "is_elastic": True,
                }
            )

        return servers
