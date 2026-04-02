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

import ray
from omegaconf import DictConfig

from verl.experimental.fully_async_policy.agent_loop import (
    FullyAsyncAgentLoopManager,
    FullyAsyncAgentLoopWorker,
)
from verl.single_controller.ray import RayWorkerGroup
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

    # ---- test helpers ----

    def set_inflight_for_test(self, server_id: str, count: int) -> None:
        """Directly set in-flight count for a server (test use only)."""
        self._inflight_requests[server_id] = count

    def is_server_removed(self, server_id: str) -> bool:
        """Return True if the server is in the removed set (test use only)."""
        return server_id in self._removed_servers

    def has_server(self, server_id: str) -> bool:
        """Return True if the server exists in the inflight table (test use only)."""
        return server_id in self._inflight_requests


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

    Extends FullyAsyncAgentLoopManager with the ability to dynamically add and
    remove hybrid (shared-GPU) rollout replicas at runtime.

    Resource model
    --------------
    ElasticRollouter manages two kinds of resources simultaneously:

    Fixed standalone replicas
        Always-on rollout servers with their own GPU pool (init_standalone).
        Created during initialisation by the base AgentLoopManager.create().
        These are the ordinary rollout workers and are never touched by the
        elastic add/remove API.

    Elastic hybrid replicas
        Share GPUs with the training engine (elastic_worker_group).
        Their RolloutReplica objects are created and initialised (init_hybrid)
        inside ElasticAgentLoopManager.create() using the provided
        elastic_worker_group.  After initialisation they are put to sleep so
        the training engine can reclaim their GPUs.
        They start in a sleeping state and are NOT added to the active LB pool
        until add_elastic_replica() is called.

    Dynamic lifecycle (hybrid replicas only)
        add_elastic_replica():    wake_up() → register with LB
        remove_elastic_replica(): abort in-flight → LB cleanup → sleep()

    Use ElasticAgentLoopManager.create() (not the base create()) so that the
    elastic replicas are created and registered during initialisation.

    Ordering guarantee for partial-rollout auto-resume on removal:
      LB mark-removing → workers remove handle → abort_all_requests → LB cleanup → sleep()
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
        # Override after super().__init__() to prevent FullyAsyncAgentLoopManager
        # from overwriting this with FullyAsyncAgentLoopWorker.  ElasticAgentLoopWorker
        # is required so that _notify_workers_server_added/removed can call
        # add_server() / remove_server() on each worker to keep their
        # _server_id_to_handle maps in sync when elastic replicas are activated.
        self.agent_loop_workers_class = ray.remote(ElasticAgentLoopWorker)
        # resource_id -> RolloutReplica  (hybrid elastic replicas only)
        self._elastic_replicas: dict[str, RolloutReplica] = {}
        # resource_id -> server_address  (for LB / worker notification)
        self._elastic_addresses: dict[str, str] = {}
        # resource_id -> param_version
        self._elastic_versions: dict[str, int] = {}

        # Pre-registered elastic replicas: bound at init time but still sleeping.
        # Keyed by resource_id; populated by create() before add_elastic_replica().
        self._registered_elastic_replicas: dict[str, RolloutReplica] = {}

        # Timing / counters
        self._total_elastic_adds: int = 0
        self._total_elastic_removes: int = 0
        self._last_elastic_add_time: float = 0.0
        self._last_elastic_remove_time: float = 0.0

    @classmethod
    async def create(
        cls,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool=None,
        reward_loop_worker_handles: list = None,
        teacher_model_manager=None,
        elastic_worker_group: RayWorkerGroup | None = None,
    ) -> "ElasticAgentLoopManager":
        """
        Create an ElasticAgentLoopManager.

        This overrides AgentLoopManager.create() to additionally create and
        register hybrid elastic replicas from elastic_worker_group.  Replicas
        are instantiated and initialised (init_hybrid) here, then put to sleep
        so the training engine can reclaim their GPUs.  add_elastic_replica()
        later wakes them up on demand.

        Fixed standalone replicas are initialised by the base class as usual
        (init_standalone when worker_group is None, init_hybrid otherwise).

        Args:
            config: Full training configuration.
            worker_group: ActorRolloutRef worker group for fixed standalone
                replicas; None means standalone mode for the fixed pool.
            rollout_resource_pool: Resource pool for hybrid fixed replicas
                (TRT-LLM colocated mode).
            reward_loop_worker_handles: Actor handles for streaming reward
                computation.
            teacher_model_manager: Manager for distillation teacher servers.
            elastic_worker_group: Worker group whose GPUs are shared with the
                training engine and will be used for elastic hybrid replicas.
                RolloutReplica objects are created and initialised via
                init_hybrid() inside this method, then immediately put to sleep.
                Pass None (default) when there are no elastic resources.

        Returns:
            Fully initialised ElasticAgentLoopManager.
        """
        instance = cls(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            teacher_model_manager=teacher_model_manager,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        # ── Step 1: elastic replicas first (replica_rank 0 … N_e-1) ──────────
        # Initialise and immediately sleep them so the training engine can
        # reclaim GPU memory.  Starting from rank 0 gives elastic actors the
        # lowest-numbered placement-group bundles which are co-located with the
        # training engine, maximising GPU affinity on multi-node deployments.
        num_elastic = 0
        if elastic_worker_group is not None:
            num_elastic = await instance._initialize_elastic_replicas(elastic_worker_group, start_rank=0)

        # ── Step 2: fixed replicas (replica_rank N_e … N_e+N_f-1) ───────────
        # start_rank=num_elastic ensures the Ray actor names (e.g.
        # sglang_server_{rank}_{node}) are globally unique and never collide
        # with the elastic actors created above.
        await instance._initialize_llm_servers(start_rank=num_elastic)

        # ── Step 3: build LB with all currently active (fixed) servers ────────
        # Elastic servers start sleeping; they are added to the LB on demand
        # via add_elastic_replica().
        await instance._init_global_load_balancer()
        await instance._init_agent_loop_workers()

        logger.info(
            f"[ElasticAgentLoopManager] Created: "
            f"{len(instance.rollout_replicas)} fixed replicas (rank {num_elastic}+), "
            f"{num_elastic} elastic replicas registered (sleeping, rank 0-{num_elastic - 1})"
        )
        return instance

    async def _initialize_elastic_replicas(
        self,
        elastic_worker_group: RayWorkerGroup,
        start_rank: int = 0,
    ) -> int:
        """
        Create, initialise (init_hybrid), and sleep elastic hybrid replicas.

        Called internally by create() when elastic_worker_group is provided.
        Each replica is assigned a contiguous slice of workers from
        elastic_worker_group, and its ``replica_rank`` starts at ``start_rank``
        so that it is globally unique across both elastic and fixed replicas
        (avoiding Ray named-actor collisions such as ``sglang_server_0_0``).

        After init_hybrid() the replica is immediately put to sleep so that
        the training engine can use the shared GPUs.  The replica is stored in
        _registered_elastic_replicas keyed by "elastic_{i}" (0-indexed within
        the elastic group) and can be activated later via add_elastic_replica().

        Args:
            elastic_worker_group: Worker group whose workers back the elastic
                hybrid replicas.  Must already be fully initialised.
            start_rank: First global ``replica_rank`` to assign.  Pass 0 so
                that elastic actors occupy the lowest-numbered Ray actor names
                and therefore the best GPU affinity in multi-node deployments.

        Returns:
            Number of elastic replicas created (so the caller can pass
            ``start_rank=num_elastic`` to ``_initialize_llm_servers``).
        """
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        num_elastic_replicas = elastic_worker_group.world_size // rollout_world_size

        elastic_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + i,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for i in range(num_elastic_replicas)
        ]

        # Initialise all replicas concurrently.
        await asyncio.gather(*[replica.init_hybrid(elastic_worker_group) for replica in elastic_replicas])

        # Immediately sleep so the training engine can reclaim GPU memory.
        await asyncio.gather(*[replica.sleep() for replica in elastic_replicas])

        # Register without activating in the LB pool.
        for i, replica in enumerate(elastic_replicas):
            resource_id = f"elastic_{i}"
            self._registered_elastic_replicas[resource_id] = replica
            logger.info(
                f"[ElasticAgentLoopManager] Elastic replica '{resource_id}' "
                f"(rank={start_rank + i}) initialised at {replica._server_address} "
                f"(sleeping, not yet in LB pool)"
            )

        logger.info(
            f"[ElasticAgentLoopManager] {num_elastic_replicas} elastic replicas "
            f"initialised (rank {start_rank}–{start_rank + num_elastic_replicas - 1}), sleeping"
        )
        return num_elastic_replicas

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
        param_version: int,
    ) -> bool:
        """
        Activate a pre-registered elastic hybrid replica and add it to the
        active server pool.

        The replica must have been pre-registered via the elastic_replicas
        argument of ElasticAgentLoopManager.create().  Its server is already
        bound to the shared resource pool (init_hybrid / init_colocated) and is
        currently sleeping.  This method:
          1. wake_up() — restores model weights / kv-cache
          2. Notifies all AgentLoopWorkers (push handle into local map)
          3. Registers the server with the load balancer (after workers are ready)

        The worker notification MUST happen before LB registration to avoid a
        race condition where the LB routes a request to the new server before
        any worker has the handle in its _server_id_to_handle map.

        Args:
            resource_id: Unique identifier of the elastic resource to activate.
            param_version: Parameter version the replica has been synced to.

        Returns:
            True on success, False on failure (e.g. not pre-registered, already active).
        """
        if resource_id in self._elastic_replicas:
            logger.warning(f"[ElasticAgentLoopManager] Replica '{resource_id}' already active, skipping")
            return False

        replica = self._registered_elastic_replicas.get(resource_id)
        if replica is None:
            logger.error(
                f"[ElasticAgentLoopManager] Replica '{resource_id}' is not registered. "
                f"Elastic replicas are created inside ElasticAgentLoopManager.create() "
                f"via elastic_worker_group; ensure that argument was provided."
            )
            return False

        server_address = replica._server_address
        server_handle = replica._server_handle

        logger.info(
            f"[ElasticAgentLoopManager] Activating elastic replica '{resource_id}' "
            f"at {server_address} (param_version={param_version})"
        )
        try:
            # 1. Wake up: restore model weights / kv-cache on the shared GPU pool.
            await replica.wake_up()
            logger.info(f"[ElasticAgentLoopManager] Replica '{resource_id}' woken up at {server_address}")

            # 2. Push the new handle into every AgentLoopWorker's local map FIRST,
            #    so they can resolve the server_id before the LB starts routing to it.
            await self._notify_workers_server_added(server_address, server_handle)

            # 3. Only after all workers have the handle, register with the load balancer
            #    so new requests can be routed to this server without racing.
            await self.global_load_balancer.add_server.remote(server_id=server_address)

            # 4. Record active state.
            self._elastic_replicas[resource_id] = replica
            self._elastic_addresses[resource_id] = server_address
            self._elastic_versions[resource_id] = param_version
            self._total_elastic_adds += 1
            self._last_elastic_add_time = time.time()

            logger.info(
                f"[ElasticAgentLoopManager] Replica '{resource_id}' added at {server_address}. "
                f"Active elastic replicas: {len(self._elastic_replicas)}"
            )
            return True

        except Exception as e:
            logger.error(f"[ElasticAgentLoopManager] Failed to activate replica '{resource_id}': {e}")
            return False

    async def remove_elastic_replica(self, resource_id: str) -> bool:
        """
        Deactivate an active elastic hybrid replica and return its GPUs to the
        training engine.

        "Removal" means sleeping the server so model weights / kv-cache are
        released and the shared GPU pool can be reclaimed by the training engine.
        The replica object is NOT destroyed; it remains in _registered_elastic_replicas
        and can be re-activated later via add_elastic_replica().

        Ordering is critical for partial-rollout auto-resume:
          1. LB: mark server as removing (no NEW requests)
          2. Workers: remove server handle BEFORE abort, so that when
             FullyAsyncLLMServerManager retries after stop_reason="aborted" the
             dead handle is already gone and traffic re-routes to healthy servers.
          3. abort_all_requests() → triggers partial-rollout resume
          4. LB: full cleanup
          5. sleep() — releases model weights / kv-cache; GPUs returned to trainer

        Args:
            resource_id: Unique identifier of the elastic resource to deactivate.

        Returns:
            True on success, False if resource_id is not currently active.
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

            # Step 5: Sleep the replica to release model weights / kv-cache so the
            #         training engine can reclaim the shared GPUs.
            try:
                await replica.sleep()
                logger.info(
                    f"[ElasticAgentLoopManager] Replica '{resource_id}' put to sleep; GPUs released for training engine"
                )
            except Exception as e:
                logger.warning(f"[ElasticAgentLoopManager] Error sleeping replica '{resource_id}': {e}")

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
