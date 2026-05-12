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
from typing import Any, Optional
from uuid import uuid4

import aiohttp
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
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers."""

    def __init__(self, servers: dict[str, ray.actor.ActorHandle], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not servers:
            raise ValueError("server must be non-empty")

        self._server = servers
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str) -> str:
        """Acquire a server for the given request, reusing the same server for multi-turn conversations."""
        # request-level sticky (multi-turn: same conversation -> same server)
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            self._inflight_requests[server_id] += 1
            return server_id

        # new request: route to least loaded server
        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes, decrementing its inflight count."""
        if server_id not in self._inflight_requests:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight_requests[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Add new servers to the server handles."""
        raise NotImplementedError("Not implemented")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Remove servers from the server handles."""
        raise NotImplementedError("Not implemented")


def _parse_token_output(data: dict) -> TokenOutput:
    """Parse a sglang /generate HTTP response dict into a TokenOutput.

    sglang response shape::

        {
          "output_ids": [101, 202, ...],
          "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[logprob, token_id, rank], ...],
            "num_preempted": 0,
            "routed_experts": null,
            "global_steps": 42,
            "input_token_logprobs": [...],
            "input_top_logprobs": [...]
          }
        }
    """
    meta_info = data.get("meta_info", {})
    token_ids = list(data.get("output_ids", []))

    log_probs = None
    output_token_logprobs = meta_info.get("output_token_logprobs") or []
    if output_token_logprobs and len(output_token_logprobs) == len(token_ids):
        log_probs = [float(lp) for lp, _, _ in output_token_logprobs]

    finish_reason = meta_info.get("finish_reason")
    stop_reason = finish_reason["type"] if finish_reason else None

    routed_experts = meta_info.get("routed_experts")
    if routed_experts is not None:
        routed_experts = torch.tensor(routed_experts)

    num_preempted = meta_info.get("num_preempted", 0)

    extra_fields: dict[str, Any] = {}
    global_steps = meta_info.get("global_steps")
    if global_steps is not None:
        extra_fields["global_steps"] = global_steps

    return TokenOutput(
        token_ids=token_ids,
        log_probs=log_probs,
        routed_experts=routed_experts,
        stop_reason=stop_reason,
        num_preempted=num_preempted,
        extra_fields=extra_fields,
    )


class LLMServerClient:
    """Proxy client for one or more sglang servers.

    Supports two routing modes selected at construction time:

    * **least_inflight** (default): routes via ``GlobalRequestLoadBalancer`` Ray actor
      using in-flight request counts + LRU sticky session.
    * **sglang_router**: routes via the sgl-model-gateway Rust Router HTTP endpoint;
      sticky session is handled by the Router's prefix-cache-aware policy.
    """

    def __init__(
        self,
        config: DictConfig,
        # least_inflight mode
        servers: Optional[dict[str, ray.actor.ActorHandle]] = None,
        load_balancer_handle: Optional[ray.actor.ActorHandle] = None,
        # sglang_router mode
        router_address: Optional[str] = None,
    ):
        self.config = config
        # sglang_router path
        self._router_address = router_address
        # least_inflight path
        self._load_balancer = load_balancer_handle
        self._server_id_to_handle: dict[str, ray.actor.ActorHandle] = servers or {}
        # aiohttp session for Router path (lazy-initialised inside async context)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # aiohttp session helpers (Router path only)
    # ------------------------------------------------------------------

    async def _get_or_create_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=2000,
                    limit_per_host=500,
                    ttl_dns_cache=300,
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=600.0),
                )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session. Call from LLMServerManager on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # least_inflight path helpers
    # ------------------------------------------------------------------

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
        handle = self._server_id_to_handle.get(server_id)
        if handle is None:
            raise RuntimeError(f"Unknown server_id returned by load balancer: {server_id}")
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        self._load_balancer.release_server.remote(server_id=server_id)

    # ------------------------------------------------------------------
    # Router path: HTTP generate
    # ------------------------------------------------------------------

    async def _generate_via_router(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        **kwargs: Any,
    ) -> TokenOutput:
        url = f"{self._router_address}/generate"
        rollout_cfg = self.config.actor_rollout_ref.rollout

        sampling_params = dict(sampling_params)  # shallow copy – do not mutate caller's dict

        # --- max_new_tokens: mirror SGLangHttpServer.generate() logic ---
        max_model_len = rollout_cfg.get("max_model_len") or 0
        max_possible_tokens = max_model_len - len(prompt_ids) - 1 if max_model_len else None
        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            max_new_tokens = min(
                rollout_cfg.response_length,
                rollout_cfg.prompt_length + rollout_cfg.response_length - len(prompt_ids),
            )
        if max_possible_tokens is not None:
            max_new_tokens = max(0, min(max_new_tokens, max_possible_tokens))
        sampling_params["max_new_tokens"] = max_new_tokens

        # --- logprobs / prompt_logprobs → sglang API fields ---
        return_logprob = sampling_params.pop("logprobs", False)
        prompt_logprobs = sampling_params.pop("prompt_logprobs", None)
        if prompt_logprobs is not None:
            return_logprob = True

        payload: dict[str, Any] = {
            "rid": uuid4().hex,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
        }
        if prompt_logprobs is not None:
            payload["logprob_start_len"] = 0
            if prompt_logprobs > 0:
                payload["top_logprobs_num"] = prompt_logprobs
        if rollout_cfg.get("enable_rollout_routing_replay"):
            payload["return_routed_experts"] = True
        # video_data not yet supported by sglang HTTP /generate
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        headers = {
            "x-verl-request-id": request_id,  # Router sticky-session key
            "Content-Type": "application/json",
        }

        logger.debug(
            "router req rid=%s fields=%s prompt_ids_len=%d max_new_tokens=%s return_logprob=%s",
            request_id, list(payload.keys()), len(prompt_ids),
            payload.get("sampling_params", {}).get("max_new_tokens"),
            payload.get("return_logprob"),
        )

        session = await self._get_or_create_session()
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
                    logger.debug(
                        "router resp rid=%s top_fields=%s meta_fields=%s output_ids_len=%d finish_reason=%s",
                        request_id, list(data.keys()),
                        list(data.get("meta_info", {}).keys()),
                        len(data.get("output_ids", [])),
                        data.get("meta_info", {}).get("finish_reason"),
                    )
                    return _parse_token_output(data)
            except aiohttp.ClientResponseError as exc:
                if exc.status < 500:
                    raise  # 4xx – caller error, do not retry
                logger.warning("Router returned HTTP %d, attempt %d/3", exc.status, attempt + 1)
                last_exc = exc
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                logger.warning("Router connection error: %s, attempt %d/3", exc, attempt + 1)
                last_exc = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"generate via Router failed after 3 attempts for request {request_id}") from last_exc

    # ------------------------------------------------------------------
    # least_inflight path: Ray RPC generate (original behaviour)
    # ------------------------------------------------------------------

    async def _generate_via_ray(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        **kwargs: Any,
    ) -> TokenOutput:
        server_id, server = await self._acquire_server(request_id)
        try:
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            request_id: Sticky-session key; multi-turn calls with the same id
                are routed to the same replica (for KV-cache reuse).
            prompt_ids: Input token ids.
            sampling_params: Sampling parameters forwarded to the engine.

        Returns:
            TokenOutput with generated token ids and optional log-probs.
        """
        if self._router_address:
            return await self._generate_via_router(
                request_id, prompt_ids, sampling_params, image_data, video_data, **kwargs
            )
        return await self._generate_via_ray(request_id, prompt_ids, sampling_params, image_data, video_data, **kwargs)


class FullyLLMServerClient(LLMServerClient):
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
        """Initialise the load-balancing layer.

        Chooses between the Rust Router sidecar (strategy=sglang_router) and the
        original GlobalRequestLoadBalancer Ray actor (strategy=least_inflight).
        Falls back to least_inflight when load_balance is absent from config.
        """
        lb_cfg = getattr(self.rollout_config, "load_balance", None)
        strategy = getattr(lb_cfg, "strategy", "least_inflight") if lb_cfg is not None else "least_inflight"
        use_router = strategy == "sglang_router" and self.rollout_config.name == "sglang"

        if use_router:
            await self._init_sglang_router(lb_cfg)
        else:
            self.global_load_balancer = GlobalRequestLoadBalancer.remote(
                servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
                max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            )
            self.router_address: Optional[str] = None
            self.router_actor = None

    async def _init_sglang_router(self, lb_cfg) -> None:
        """Start a sgl-model-gateway Rust Router and register all replica addresses."""
        from verl.workers.rollout.sglang_router_actor import SGLangRouterActor

        router_cfg = getattr(lb_cfg, "router", None)

        actor_options: dict[str, Any] = {"num_cpus": 1}

        # In STANDALONE mode pin the Router to the first rollout node to minimise
        # cross-node hops for the generate path.
        if not self.worker_group:
            try:
                rollout_node_id = await self.server_handles[0].get_node_id.remote()
                actor_options["scheduling_strategy"] = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=rollout_node_id, soft=True
                )
            except Exception:
                pass  # best-effort placement; don't fail startup

        worker_urls = [
            addr if addr.startswith(("http://", "https://")) else f"http://{addr}" for addr in self.server_addresses
        ]
        self.router_actor = SGLangRouterActor.options(**actor_options).remote(
            worker_urls=worker_urls,
            policy=getattr(router_cfg, "policy", "cache_aware") if router_cfg else "cache_aware",
            cache_threshold=getattr(router_cfg, "cache_threshold", 0.3) if router_cfg else 0.3,
            balance_abs_threshold=getattr(router_cfg, "balance_abs_threshold", 64) if router_cfg else 64,
            balance_rel_threshold=getattr(router_cfg, "balance_rel_threshold", 1.5) if router_cfg else 1.5,
            request_id_headers=["x-verl-request-id"],
            health_check_interval_secs=getattr(router_cfg, "health_check_interval_secs", 60) if router_cfg else 60,
        )
        self.router_address = await self.router_actor.start_and_wait.remote(timeout=120.0)
        print(f"LLMServerManager: Rust Router started at {self.router_address}", flush=True)
        self.global_load_balancer = None
        logger.info("LLMServerManager: Rust Router started at %s", self.router_address)

    async def router_sleep(self) -> None:
        """No-op: workers stay registered; Router health checks are disabled."""

    async def router_wake_up(self) -> None:
        """No-op: workers were registered at Router startup and stay permanently."""

    def get_client(self, fully_async: bool = False) -> LLMServerClient:
        """Return an LLMServerClient configured for the active routing mode.

        Args:
            fully_async: When True, returns a FullyLLMServerClient that handles
                partial-rollout resume transparently.
        """
        client_cls = FullyLLMServerClient if fully_async else LLMServerClient

        if getattr(self, "router_address", None):
            # sglang_router path: generate via Rust Router HTTP endpoint
            return client_cls(
                config=self.config,
                router_address=self.router_address,
            )

        # least_inflight path: original Ray-actor-based routing
        servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
        return client_cls(
            config=self.config,
            servers=servers,
            load_balancer_handle=self.global_load_balancer,
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

