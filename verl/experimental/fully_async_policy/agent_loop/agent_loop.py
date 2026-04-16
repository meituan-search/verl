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
import asyncio
import logging
import os
import time
from typing import Any, Optional

import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    AsyncLLMServerManager,
    TokenOutput,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.teacher_loop import TeacherModelManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import (
    rollout_trace_op,
)
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout import RolloutReplica
from verl.workers.rollout.replica import RolloutMode

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class ElasticGlobalRequestLoadBalancer:
    """
    Simplified GlobalRequestLoadBalancer with dynamic add/remove support.

    Design principles:
    - Round-robin / least-loaded routing among active servers only.
    - No sticky session cache — avoids stale routing after server removal.
    - No two-phase removal (remove_server → cleanup_removed_server) —
      just add_server / remove_server directly.  Caller is responsible for
      draining in-flight requests *before* calling remove_server.
    - All operations are O(1) and stateless enough to avoid deadlocks.
    """

    def __init__(self, server_actor_ids: list[str], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")

        # Active servers with their in-flight request counts.
        # Only servers in this dict are eligible for acquire_server().
        self._servers: dict[str, int] = {sid: 0 for sid in server_actor_ids}

    def acquire_server(self, request_id: str) -> str:
        """Acquire a server for the given request (least-loaded)."""
        if not self._servers:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(self._servers, key=self._servers.get)
        self._servers[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id in self._servers and self._servers[server_id] > 0:
            self._servers[server_id] -= 1

    def add_server(self, server_id: str) -> None:
        """Add a new server to the pool.  No-op if already present."""
        if server_id not in self._servers:
            self._servers[server_id] = 0
            print(f"[ElasticLB] Added server: {server_id}")

    def remove_server(self, server_id: str) -> None:
        """Remove a server from the pool immediately.

        The caller MUST ensure no in-flight requests are routed to this
        server before calling this (e.g. by aborting requests and removing
        the handle from all workers' server_manager maps).
        """
        self._servers.pop(server_id, None)
        print(f"[ElasticLB] Removed server: {server_id}")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._servers.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._servers.keys())


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    """FullyAsyncLLMServerManager supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            if output.routed_experts is not None:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat([final_output.routed_experts, output.routed_experts], dim=0)
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            await asyncio.sleep(10)

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


class FullyAsyncAgentLoopWorker(AgentLoopWorker):
    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        teacher_servers: list[tuple[str, ray.actor.ActorHandle]] = None,
        teacher_load_balancer_handle: ray.actor.ActorHandle = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.server_manager = FullyAsyncLLMServerManager(config, servers, load_balancer_handle)
        super().__init__(
            config,
            servers,
            load_balancer_handle,
            teacher_servers,
            teacher_load_balancer_handle,
            reward_loop_worker_handles,
        )

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


class FullyAsyncAgentLoopManager(AgentLoopManager):
    """Unified AgentLoopManager for fully async training with elastic replica support.

    Manages two categories of rollout replicas:

    * **Fixed replicas** — always-on rollout servers backed by ``worker_group``
      (the rollouter's own GPU pool).  Created during :meth:`create` via
      ``init_standalone()`` and registered with the
      :class:`ElasticGlobalRequestLoadBalancer` at startup.  These are never
      touched by the elastic add/remove API.

    * **Elastic hybrid replicas** — optional rollout servers that **share GPUs**
      with the training engine (``elastic_worker_group``, injected by the
      caller before ``init_workers()``).  Created via ``init_hybrid()`` inside
      :meth:`create`, immediately put to **sleep** so the training engine can
      reclaim their GPU memory, and activated on demand.

    Internal data structures::

        self.rollout_replicas: list[RolloutReplica]   # fixed + active elastic
        self.elastic_replicas: dict[str, RolloutReplica]  # all elastic (sleeping + alive), keyed by "elastic_{i}"
        self.alive_replicas: dict[str, RolloutReplica]    # currently active (awake + in LB) subset
        self.alive_addresses: dict[str, str]               # resource_id → server_address for alive ones

    Elastic replica lifecycle::

        create()                          # init_hybrid() on trainer GPUs
          ↓
        sleep()                           # release weights/kv-cache; trainer owns GPUs now
          ↓  ────────────────────────────────────────────────────────
        add_elastic_replica(resource_id)  # called by Trainer via RPC → Rollouter
          ├─ wake_up()                   # restore model weights / kv-cache
          ├─ _notify_workers_server_added()  # push handle into every worker's map
          └─ global_load_balancer.add_server()  # start routing requests
          ↓
        [serving generation requests as part of rollout_replicas]
          ↓
        remove_elastic_replica(resource_id)
          ├─ global_load_balancer.remove_server()    # stop NEW requests
          ├─ _notify_workers_server_removed()       # drop handle from workers
          ├─ abort_all_requests()                   # trigger partial-rollout auto-resume
          ├─ global_load_balancer.cleanup_removed_server()
          └─ sleep()                                # release GPUs back to trainer

    The ordering in ``remove_elastic_replica`` is critical: workers must drop
    the server handle **before** ``abort_all_requests()`` is called, so that
    when :class:`FullyAsyncLLMServerManager` retries after receiving
    ``stop_reason="aborted"`` the dead handle is already gone and traffic is
    re-routed to healthy servers (partial-rollout resume).
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        teacher_model_manager: TeacherModelManager = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.agent_loop_workers_class = ray.remote(FullyAsyncAgentLoopWorker)
        super().__init__(config, worker_group, rollout_resource_pool, teacher_model_manager, reward_loop_worker_handles)
        if self.distillation_enabled:
            raise NotImplementedError("Distillation is not implemented in FullyAsyncAgentLoopManager yet.")

        # Rollout replicas (alive hybrid and fixed)
        self.rollout_replicas = []
        # Pre-registered elastic replicas: bound at init time but still sleeping.
        # Keyed by resource_id; populated by create() before add_elastic_replica().
        self.elastic_replicas: dict[str, RolloutReplica] = {}
        # resource_id -> RolloutReplica  (hybrid elastic replicas only)
        self.alive_replicas: dict[str, RolloutReplica] = {}
        # resource_id -> server_address  (for LB / worker notification)
        self.alive_addresses: dict[str, str] = {}
        # Prometheus server addresses
        self.prometheus_server_addresses = []

        # Timing / counters
        self.total_elastic_adds: int = 0
        self.total_elastic_removes: int = 0
        self.last_elastic_add_time: float = 0.0
        self.last_elastic_remove_time: float = 0.0

    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool=None,
        reward_loop_worker_handles: list = None,
        teacher_model_manager=None,
        elastic_worker_group: RayWorkerGroup = None,
    ) -> "FullyAsyncAgentLoopManager":
        """
        Create a FullyAsyncAgentLoopManager with both fixed and optional elastic replicas.

        Initialisation proceeds in three steps:

        1. **Elastic replicas** (``elastic_worker_group``): if provided, create
           ``RolloutReplica`` objects via ``init_hybrid()`` on the trainer's GPUs
           (``replica_rank 0 … N_e-1``), register them in ``elastic_replicas``,
           and immediately ``sleep()`` each one so the training engine can reclaim
           GPU memory.  Rank 0 ensures elastic actors get the lowest-numbered
           placement-group bundles for best GPU affinity on multi-node deployments.

        2. **Fixed replicas** (``worker_group``): create standalone rollout replicas
           with ``replica_rank starting at N_e`` to avoid Ray named-actor collisions
           (e.g. ``sglang_server_0_0``) with the elastic actors above.
           These are registered with the load balancer at startup.

        3. **Infrastructure**: build the :class:`ElasticGlobalRequestLoadBalancer`
           with fixed server addresses, then spawn :class:`FullyAsyncAgentLoopWorker`
           actors.  Elastic servers are **not** added to the LB; they join on demand
           via :meth:`add_elastic_replica`.

        Args:
            config: Full training configuration.
            worker_group: Worker group for fixed rollout replicas.
            rollout_resource_pool: Resource pool for hybrid fixed replicas.
            reward_loop_worker_handles: Actor handles for streaming reward
                computation.
            teacher_model_manager: Manager for distillation teacher servers.
            elastic_worker_group: Worker group whose GPUs are shared with the
                training engine (typically the trainer's ``actor_rollout_wg``).
                Pass ``None`` (default) when there are no elastic resources.

        Returns:
            Fully initialised FullyAsyncAgentLoopManager.
        """
        instance = cls(config, worker_group, rollout_resource_pool, teacher_model_manager, reward_loop_worker_handles)

        # ── Step 1: elastic replicas first (replica_rank 0 … N_e-1) ──────────
        # Initialise and immediately sleep them so the training engine can
        # reclaim GPU memory.  Starting from rank 0 gives elastic actors the
        # lowest-numbered placement-group bundles which are co-located with the
        # training engine, maximising GPU affinity on multi-node deployments.
        num_elastic = 0
        if elastic_worker_group is not None:
            num_elastic = await instance._initialize_elastic_replicas(start_rank=0, worker_group=elastic_worker_group)

        # ── Step 2: fixed replicas (replica_rank N_e … N_e+N_f-1) ───────────
        # start_rank=num_elastic ensures the Ray actor names (e.g.
        # sglang_server_{rank}_{node}) are globally unique and never collide
        # with the elastic actors created above.
        num_elastic = await instance._initialize_elastic_replicas(start_rank=num_elastic)

        # ── Step 3: build LB with all currently active (fixed) servers ────────
        # Elastic servers start sleeping; they are added to the LB on demand
        # via add_elastic_replica().
        await instance._init_global_load_balancer()
        await instance._init_agent_loop_workers()

        print(
            f"[FullyAsyncAgentLoopManager] Created: "
            f"{len(instance.rollout_replicas)} fixed replicas (rank {num_elastic}+), "
            f"{num_elastic} elastic replicas registered (sleeping, rank 0-{num_elastic - 1})"
        )
        return instance

    async def _initialize_elastic_replicas(
        self,
        start_rank: int = 0,
        worker_group: RayWorkerGroup = None,
    ) -> int:
        """
        Create, initialise (init_hybrid), and sleep elastic hybrid replicas.

        Called internally by create() when elastic_worker_group is provided.
        Each replica is assigned a contiguous slice of workers from
        elastic_worker_group, and its ``replica_rank`` starts at ``start_rank``
        so that it is globally unique across both elastic and fixed replicas
        (avoiding Ray named-actor collisions such as ``sglang_server_0_0``).

        After init_hybrid() the replica is immediately put to sleep so that
        The replica is stored in
        elastic_replicas keyed by "elastic_{i}" (0-indexed within
        the elastic group) and can be activated later via add_elastic_replica().

        Args:
            worker_group: Worker group whose workers back the elastic
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

        world_size = (
            worker_group.world_size
            if worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )

        num_replicas = world_size // rollout_world_size

        tmp_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + i,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for i in range(num_replicas)
        ]

        if worker_group:
            await asyncio.gather(*[replica.init_hybrid(worker_group) for replica in tmp_replicas])
            # Register elastic replicas.
            for i, replica in enumerate(tmp_replicas):
                resource_id = f"elastic_{i}"
                self.elastic_replicas[resource_id] = replica
                print(
                    f"[FullyAsyncAgentLoopManager] Elastic replica '{resource_id}' "
                    f"(rank={start_rank + i}) initialised at {replica._server_address} "
                )
            elastic_addresses = [replica._server_address for replica in tmp_replicas]
            self.prometheus_server_addresses.extend(elastic_addresses)
            print(f"AgentLoopManager Elastic: {elastic_addresses}")
        else:
            self.rollout_replicas = tmp_replicas
            await asyncio.gather(*[replica.init_standalone() for replica in self.rollout_replicas])
            self.server_handles = [replica._server_handle for replica in self.rollout_replicas]
            self.server_addresses = [replica._server_address for replica in self.rollout_replicas]
            self.prometheus_server_addresses.extend(self.server_addresses)
            print(f"AgentLoopManager Standalone: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            print(f"Prometheus: {self.prometheus_server_addresses}")
            update_prometheus_config(
                self.rollout_config.prometheus, self.prometheus_server_addresses, self.rollout_config.name
            )

        return num_replicas

    async def _init_global_load_balancer(self) -> None:
        """Override to use ElasticGlobalRequestLoadBalancer (supports add/remove)."""
        self.global_load_balancer = ray.remote(ElasticGlobalRequestLoadBalancer).remote(
            server_actor_ids=self.server_addresses,
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        )
        print(f"[FullyAsyncAgentLoopManager] Load balancer initialised with {len(self.server_addresses)} servers")

    async def add_elastic_replica(self, resource_id: str) -> bool:
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

        Returns:
            True on success, False on failure (e.g. not pre-registered, already active).
        """
        if resource_id in self.alive_replicas:
            logger.warning(f"[FullyAsyncAgentLoopManager] Replica '{resource_id}' already active, skipping")
            return False

        replica = self.elastic_replicas.get(resource_id)
        if replica is None:
            logger.error(
                f"[FullyAsyncAgentLoopManager] Replica '{resource_id}' is not registered. "
                f"Elastic replicas are created inside FullyAsyncAgentLoopManager.create() "
                f"via elastic_worker_group; ensure that argument was provided."
            )
            return False

        server_address = replica._server_address
        server_handle = replica._server_handle

        print(f"[FullyAsyncAgentLoopManager] Activating elastic replica '{resource_id}' at {server_address}")
        try:
            # 1. Wake up: restore model weights / kv-cache on the shared GPU pool.
            #    For HYBRID mode replicas, wake_up() is NOT supported because the
            #    rollout engine is woken up inside update_weights() (which performs
            #    the full rollout_mode transition: resume weights → sync → offload actor
            #    → resume kv_cache).  The caller (_trainer_side_validate) already
            #    invoked update_weights() in Phase 1 Step 1, so we skip wake_up()
            #    here and only do LB registration for HYBRID replicas.
            print(f"[FullyAsyncAgentLoopManager] add_elastic_replica('{resource_id}'): Step 1 wake_up (rollout_mode={replica.rollout_mode})")
            if replica.rollout_mode != RolloutMode.HYBRID:
                await replica.wake_up()
                print(f"[FullyAsyncAgentLoopManager] Replica '{resource_id}' woken up at {server_address}")
            else:
                print(
                    f"[FullyAsyncAgentLoopManager] Replica '{resource_id}' is HYBRID mode — "
                    f"skipping wake_up() (rollout was activated by update_weights() in Phase 1)"
                )

            # 2. Push the new handle into every AgentLoopWorker's local map FIRST,
            #    so they can resolve the server_id before the LB starts routing to it.
            print(f"[FullyAsyncAgentLoopManager] add_elastic_replica('{resource_id}'): Step 2 notify workers")
            await self._notify_workers_server_added(server_address, server_handle)

            # 3. Only after all workers have the handle, register with the load balancer
            #    so new requests can be routed to this server without racing.
            print(f"[FullyAsyncAgentLoopManager] add_elastic_replica('{resource_id}'): Step 3 LB add_server")
            await self.global_load_balancer.add_server.remote(server_id=server_address)

            # 4. Record active state.
            self.alive_replicas[resource_id] = replica
            self.alive_addresses[resource_id] = server_address
            self.rollout_replicas.append(replica)
            self.total_elastic_adds += 1
            self.last_elastic_add_time = time.time()

            print(
                f"[FullyAsyncAgentLoopManager] Replica '{resource_id}' added at {server_address}. "
                f"Active elastic replicas: {len(self.alive_replicas)}"
            )
            return True

        except Exception as e:
            logger.error(f"[FullyAsyncAgentLoopManager] Failed to activate replica '{resource_id}': {e}")
            return False

    async def remove_elastic_replica(self, resource_id: str) -> bool:
        """
        Deactivate an active elastic hybrid replica and return its GPUs to the
        training engine.

        "Removal" means sleeping the server so model weights / kv-cache are
        released and the shared GPU pool can be reclaimed by the training engine.
        The replica object is NOT destroyed; it remains in elastic_replicas
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
        if resource_id not in self.alive_replicas:
            logger.warning(f"[FullyAsyncAgentLoopManager] Replica {resource_id} not found, skipping")
            return False

        server_address = self.alive_addresses[resource_id]
        replica = self.alive_replicas[resource_id]

        print(f"[FullyAsyncAgentLoopManager] Removing elastic replica {resource_id} at {server_address}")
        try:
            # Step 1: Stop new routing to this server
            print(f"[FullyAsyncAgentLoopManager] remove_elastic('{resource_id}'): Step 1 LB remove_server")
            await self.global_load_balancer.remove_server.remote(server_id=server_address)

            # Step 2: Remove handle from workers BEFORE aborting
            print(f"[FullyAsyncAgentLoopManager] remove_elastic('{resource_id}'): Step 2 notify workers server_removed")
            await self._notify_workers_server_removed(server_address)

            # Step 3: Abort in-flight requests → triggers partial-rollout resume
            print(f"[FullyAsyncAgentLoopManager] remove_elastic('{resource_id}'): Step 3 abort_all_requests")
            await replica.abort_all_requests()
            print(
                f"[FullyAsyncAgentLoopManager] Aborted in-flight requests on {server_address}; "
                f"partial-rollout resume will redirect them to healthy servers"
            )

            # Step 4: (Skipped) ElasticGlobalRequestLoadBalancer uses immediate remove_server()
            #         (no two-phase cleanup).  The server was already removed from the LB
            #         in Step 1, so there is nothing to clean up here.

            # Step 5: Sleep the replica to release model weights / kv-cache so the
            #         training engine can reclaim the shared GPUs.
            print(f"[FullyAsyncAgentLoopManager] remove_elastic('{resource_id}'): Step 5 sleep (release KV cache + weights)")
            await replica.sleep()
            print(f"[FullyAsyncAgentLoopManager] remove_elastic('{resource_id}'): Step 5 sleep DONE")
            print(
                f"[FullyAsyncAgentLoopManager] Replica '{resource_id}' put to sleep; GPUs released for training engine"
            )
            # Remove tracking state
            self.alive_replicas.pop(resource_id)
            self.alive_addresses.pop(resource_id)
            if replica in self.rollout_replicas:
                self.rollout_replicas.remove(replica)
            self.total_elastic_removes += 1
            self.last_elastic_remove_time = time.time()

            print(
                f"[FullyAsyncAgentLoopManager] Replica {resource_id} removed. Total elastic: {len(self.alive_replicas)}"
            )
            return True

        except Exception as e:
            logger.error(f"[FullyAsyncAgentLoopManager] Failed to remove replica {resource_id}: {e}")
            return False

    async def force_remove_elastic_tracking(self, resource_id: str) -> None:
        """
        Force-remove elastic replica tracking state without waiting for
        abort/sleep to complete.  Used as a safety net when
        remove_elastic_replica() times out (e.g. due to stuck in-flight
        requests on the SGLang server).

        This method only cleans up in-memory tracking state (alive_replicas,
        alive_addresses, rollout_replicas, server_addresses/handles, LB).
        It does NOT call abort_all_requests() or sleep() — the caller has
        already decided those are stuck / timed out.
        """
        print(f"[FullyAsyncAgentLoopManager] force_remove_elastic_tracking('{resource_id}'): forcing cleanup")

        if resource_id not in self.alive_replicas:
            logger.warning(
                f"[FullyAsyncAgentLoopManager] force_remove_elastic_tracking: "
                f"'{resource_id}' not in alive_replicas, nothing to do"
            )
            return

        server_address = self.alive_addresses.get(resource_id)
        replica = self.alive_replicas[resource_id]

        try:
            # 1. Ensure server is removed from LB (idempotent)
            if server_address:
                try:
                    await self.global_load_balancer.remove_server.remote(server_id=server_address)
                except Exception:
                    pass  # May already be removed or actor is dead

            # 2. Remove handle from workers
            if server_address:
                try:
                    await self._notify_workers_server_removed(server_address)
                except Exception as e:
                    logger.warning(f"[FullyAsyncAgentLoopManager] force_remove: worker notify failed: {e}")

            # 3. Clean up tracking state
            self.alive_replicas.pop(resource_id, None)
            self.alive_addresses.pop(resource_id, None)
            if replica in self.rollout_replicas:
                self.rollout_replicas.remove(replica)

            print(
                f"[FullyAsyncAgentLoopManager] force_remove_elastic_tracking('{resource_id}'): done. "
                f"Remaining elastic: {len(self.alive_replicas)}"
            )

        except Exception as e:
            logger.error(f"[FullyAsyncAgentLoopManager] force_remove_elastic_tracking failed: {e}")

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
                logger.debug("[FullyAsyncAgentLoopManager] Worker doesn't support add_server()")
        if futures:
            results = await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in futures],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"[FullyAsyncAgentLoopManager] Worker add_server failed: {result}")

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
                logger.debug("[FullyAsyncAgentLoopManager] Worker doesn't support remove_server()")
        if futures:
            await asyncio.gather(
                *[asyncio.wrap_future(f.future()) for f in futures],
                return_exceptions=True,
            )

    # -------------------------------------------------------------------------
    # Statistics / introspection
    # -------------------------------------------------------------------------

    def get_num_elastic_replicas(self) -> int:
        """Return the number of currently active elastic replicas."""
        return len(self.alive_replicas)

    def get_elastic_replicas_info(self) -> list[dict]:
        """Return metadata for all active elastic replicas."""
        return [
            {
                "resource_id": rid,
                "server_address": self.alive_addresses.get(rid, "unknown"),
            }
            for rid in self.alive_replicas
        ]

    def get_elastic_statistics(self) -> dict:
        """Return elastic-specific counters for monitoring."""
        return {
            "elastic/num_elastic_replicas": len(self.alive_replicas),
            "elastic/total_adds": self.total_elastic_adds,
            "elastic/total_removes": self.total_elastic_removes,
            "elastic/last_add_time": self.last_elastic_add_time,
            "elastic/last_remove_time": self.last_elastic_remove_time,
        }

    def get_active_server_count(self) -> int:
        """Total active rollout servers (fixed + elastic)."""
        fixed = len(self.rollout_replicas)
        return fixed + len(self.alive_replicas)

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
        for rid, address in self.alive_addresses.items():
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
    # Inherited / overridden from AgentLoopManager
    # -------------------------------------------------------------------------

    @auto_await
    async def generate_sequences_single(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers."""
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(prompts)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker
