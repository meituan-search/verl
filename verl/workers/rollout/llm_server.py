# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for elastic scaling.
    """

    def __init__(self, servers: dict[str, ray.actor.ActorHandle], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not servers:
            raise ValueError("servers must be non-empty")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_server(self, server_id: str, handle: ray.actor.ActorHandle) -> None:
        """Add a new server to the load balancer pool.

        Args:
            server_id: Server identifier (typically the server address).
            handle: Optional actor handle.  If provided, also registers it
                in the internal handle table
        """
        if server_id in self._inflight_requests:
            # Still update the handle if a newer one is supplied.
            if handle is not None:
                self._servers[server_id] = handle
            return
        self._inflight_requests[server_id] = 0
        self._servers[server_id] = handle
        logger.info(f"[GlobalLoadBalancer] Added server: {server_id}")

    def remove_server(self, server_id: str) -> None:
        """Remove a server from the load balancer pool.

        Immediately removes the server.  Also removes the
        associated handle from the internal registry.  In-flight requests
        that finish after removal will be silently ignored by
        :meth:`release_server`.
        """
        self._inflight_requests.pop(server_id, None)
        self._servers.pop(server_id, None)
        logger.info(f"[GlobalLoadBalancer] Removed server: {server_id}")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching

    Server handles are obtained atomically from ``GlobalRequestLoadBalancer``
    (which merges the former ``ServerHandleRegistry``), so acquire is a single
    Ray RPC — no TOCTOU race.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry.
        """
        self.config = config
        self._load_balancer = load_balancer_handle

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        return await self._load_balancer.acquire_server.remote(request_id=request_id)

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        self._load_balancer.release_server.remote(server_id=server_id)

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        server_id, server = await self._acquire_server(request_id)
        try:
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
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
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
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

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break

            await asyncio.sleep(1)

        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self):
        """Initialize the LLM server replicas."""
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    async def _init_global_load_balancer(self) -> None:
        server_map = dict(zip(self.server_addresses, self.server_handles, strict=True))
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(
            servers=server_map,
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        )

    def get_client(self, client_cls=LLMServerClient, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate (default: ``LLMServerClient``).
                Pass ``FullyAsyncLLMServerClient`` for abort-resume support.
            *args, **kwargs: Forwarded to the client constructor.
        """
        return client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])


class FullyAsyncLLMServerManager(LLMServerManager):
    """Extension of :class:`LLMServerManager` for fully async training with elastic scaling.

    Elastic replica lifecycle is managed here via :meth:`add_replica` /
    :meth:`remove_replica`, which atomically update the
    ``GlobalRequestLoadBalancer`` (which also holds the handle registry) —
    **no per-client/worker notification needed**.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        super().__init__(config, worker_group, rollout_resource_pool)
        # Pre-registered elastic replicas: bound at init time but still sleeping.
        # Keyed by resource_id; populated during _initialize_llm_servers().
        self.elastic_replicas: dict[str, RolloutReplica] = {}
        # Currently active (awake + in LB) subset of elastic replicas.
        self.alive_replicas: dict[str, RolloutReplica] = {}
        # resource_id → server_address for alive elastic replicas.
        self.alive_addresses: dict[str, str] = {}
        # Prometheus server addresses
        self.prometheus_server_addresses = []

        # Timing / counters
        self.last_elastic_add_time: float = 0.0
        self.last_elastic_remove_time: float = 0.0

    async def _initialize_llm_servers(self):
        # ── Step 1: elastic replicas first (replica_rank 0 … N_e-1) ──────────
        # Initialise and immediately sleep them so the training engine can
        # reclaim GPU memory.  Starting from rank 0 gives elastic actors the
        # lowest-numbered placement-group bundles which are co-located with the
        # training engine, maximising GPU affinity on multi-node deployments.
        num_elastic = 0
        if self.worker_group is not None:
            num_elastic = await self._initialize_replicas(start_rank=0, worker_group=self.worker_group)

        # ── Step 2: fixed replicas (replica_rank N_e … N_e+N_f-1) ───────────
        # start_rank=num_elastic ensures the Ray actor names (e.g.
        # server_{rank}_{node}) are globally unique and never collide
        # with the elastic actors created above.
        await self._initialize_replicas(start_rank=num_elastic)

        print(
            f"[FullyAsyncLLMServerManager] Created: "
            f"{len(self.rollout_replicas)} fixed replicas (rank {num_elastic}+), "
            f"{num_elastic} elastic replicas registered (sleeping, rank 0-{num_elastic - 1})"
        )

    async def _initialize_replicas(
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
        the elastic group) and can be activated later via add_replica().

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

    async def add_replica(self, resource_id: str) -> bool:
        """Activate a pre-registered elastic hybrid replica.

        Atomically registers the handle **and** adds the server to the load
        balancer in a single ``add_server(handle=…)`` call (the LB now holds
        the merged registry).  No per-client / per-worker notification required.
        """
        if resource_id in self.alive_replicas:
            logger.warning("[FullyAsyncLLMServerManager] Replica '%s' already active, skipping", resource_id)
            return False

        replica = self.elastic_replicas.get(resource_id)
        if replica is None:
            logger.error(
                "[FullyAsyncLLMServerManager] Replica '%s' is not registered. "
                "Elastic replicas are created inside FullyAsyncLLMServerManager.create() "
                "via worker_group; ensure that argument was provided.",
                resource_id,
            )
            return False

        server_address = replica._server_address
        server_handle = replica._server_handle

        try:
            # Single atomic RPC: register handle + add to LB pool.
            await self.global_load_balancer.add_server.remote(server_id=server_address, handle=server_handle)
            # Also track locally for introspection / Prometheus.
            if server_address not in self.server_addresses:
                self.server_handles.append(server_handle)
                self.server_addresses.append(server_address)
            if replica not in self.rollout_replicas:
                self.rollout_replicas.append(replica)

            self.alive_replicas[resource_id] = replica
            self.alive_addresses[resource_id] = server_address
            self.last_elastic_add_time = time.time()

            print(
                f"[FullyAsyncLLMServerManager] Replica '{resource_id}' added at {server_address}. "
                f"Active elastic replicas: {len(self.alive_replicas)}"
            )
            return True

        except Exception as e:
            logger.error("[FullyAsyncLLMServerManager] Failed to activate replica '%s': %s", resource_id, e)
            return False

    async def remove_replica(self, resource_id: str) -> bool:
        """Deactivate an active elastic hybrid replica.

        A single ``remove_server()`` call atomically removes the server from
        the LB pool **and** purges its handle from the internal registry
        (no separate deregister RPC needed).

        Note: The actual ``abort_all_requests()`` and ``sleep()`` are called by
        the Trainer AFTER this method returns.
        """
        if resource_id not in self.alive_replicas:
            logger.warning("[FullyAsyncLLMServerManager] Replica %s not found, skipping", resource_id)
            return False

        server_address = self.alive_addresses[resource_id]
        replica = self.alive_replicas[resource_id]

        try:
            # Single atomic RPC: remove from LB pool + purge handle registry.
            await self.global_load_balancer.remove_server.remote(server_id=server_address)
            # Clean up local tracking lists.
            if server_address in self.server_addresses:
                idx = self.server_addresses.index(server_address)
                self.server_addresses.pop(idx)
                self.server_handles.pop(idx)
            if replica in self.rollout_replicas:
                self.rollout_replicas.remove(replica)

            self.alive_replicas.pop(resource_id)
            self.alive_addresses.pop(resource_id)
            self.last_elastic_remove_time = time.time()

            print(
                f"[FullyAsyncLLMServerManager] Replica {resource_id} at {server_address} removed. "
                f"Total elastic: {len(self.alive_replicas)}"
            )
            return True

        except Exception as e:
            logger.error("[FullyAsyncLLMServerManager] Failed to remove replica %s: %s", resource_id, e)
            return False

    # -------------------------------------------------------------------------
    # Statistics / introspection
    # -------------------------------------------------------------------------
    def get_num_elastic_replicas(self) -> int:
        """Return the number of currently active elastic replicas."""
        return len(self.alive_replicas)

    def get_elastic_replicas_info(self) -> list[dict]:
        """Return metadata for all active elastic replicas."""
        return [{"resource_id": rid, "server_address": addr} for rid, addr in self.alive_addresses.items()]

    def get_elastic_statistics(self) -> dict:
        """Return elastic-specific counters for monitoring."""
        return {
            "elastic/num_elastic_replicas": len(self.alive_replicas),
            "elastic/last_add_time": self.last_elastic_add_time,
            "elastic/last_remove_time": self.last_elastic_remove_time,
        }

    def get_active_server_count(self) -> int:
        """Total active rollout servers (fixed + elastic)."""
        return len(self.rollout_replicas) + len(self.alive_replicas)

    def get_server_info(self) -> list[dict]:
        """Metadata for all active rollout servers."""
        servers = [
            {"address": getattr(r, "_server_address", "unknown"), "type": "fixed", "is_elastic": False}
            for r in self.rollout_replicas
        ]
        for rid, address in self.alive_addresses.items():
            servers.append({"address": address, "resource_id": rid, "type": "elastic", "is_elastic": True})
        return servers
