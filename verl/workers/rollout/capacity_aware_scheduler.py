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
"""CapacityAwareScheduler: KV-cache-aware load balancer for RL rollout replicas."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Protocol

import aiohttp
import ray

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class LoadMetrics:
    token_usage: float  # 0.0–1.0, KV cache utilization
    num_total_tokens: int  # total KV cache capacity in tokens; 0 = unknown
    num_requests_running: int = 0  # vLLM: requests currently being decoded
    num_requests_waiting: int = 0  # vLLM: requests queued, not yet scheduled


@dataclass
class ReplicaState:
    server_id: str  # also the HTTP base URL, e.g. "http://host:8000"
    handle: ray.actor.ActorHandle
    token_usage: float  # 0.0–1.0, last polled KV cache utilization
    healthy: bool
    last_polled: float  # Unix timestamp
    num_total_tokens: int = 0  # KV cache capacity; updated each poll cycle; 0 = unknown
    pending_tokens: int = 0  # tokens dispatched since last poll, not yet reflected in token_usage
    num_requests_running: int = 0  # vLLM: requests currently being decoded
    num_requests_waiting: int = 0  # vLLM: requests queued, not yet scheduled


class LoadBackend(Protocol):
    async def fetch(self, address: str) -> LoadMetrics:
        """Return current load metrics for the replica at address."""
        ...


class SGLangLoadBackend:
    """Fetches load metrics from sglang.

    Tries /v1/loads first (sglang >= 0.4); falls back to /get_load for older versions.
    /get_load returns: [{"num_reqs", "num_waiting_reqs", "num_tokens", ...}]
    /v1/loads returns: {"loads": [...], "aggregate": {"total_tokens", "total_used_tokens", ...}}
    """

    async def fetch(self, address: str) -> LoadMetrics:
        timeout = aiohttp.ClientTimeout(total=0.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{address}/v1/loads") as resp:
                if resp.status != 404:
                    resp.raise_for_status()
                    data = await resp.json()
                    agg = data.get("aggregate", {})
                    total = agg.get("total_tokens", 0)
                    used = agg.get("total_used_tokens", 0)
                    usage = used / total if total > 0 else agg.get("avg_token_usage", 0.0)
                    return LoadMetrics(token_usage=usage, num_total_tokens=total)

            # Fallback: /get_load (deprecated shim in new sglang; native in old sglang)
            # Fields: num_reqs, num_waiting_reqs, num_tokens (used+pending),
            #         num_pending_tokens (remaining capacity)
            async with session.get(f"{address}/get_load") as resp:
                resp.raise_for_status()
                data = await resp.json()
                total_reqs = sum(d.get("num_reqs", 0) for d in data)
                total_waiting = sum(d.get("num_waiting_reqs", 0) for d in data)
                used = sum(d.get("num_tokens", 0) for d in data)
                pending = sum(d.get("num_pending_tokens", 0) for d in data)
                total = used + pending
                usage = used / total if total > 0 else 0.0
                return LoadMetrics(
                    token_usage=usage,
                    num_total_tokens=total,
                    num_requests_running=total_reqs,
                    num_requests_waiting=total_waiting,
                )


class VLLMLoadBackend:
    """Fetches load metrics from vLLM Prometheus /metrics endpoint.

    num_total_tokens is derived once from vllm:cache_config_info labels:
        num_total_tokens = num_gpu_blocks * block_size
    This enables pending_tokens estimation between poll cycles.
    """

    _CACHE_METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    _RE_NUM_GPU_BLOCKS = re.compile(r'num_gpu_blocks="(\d+)"')
    _RE_BLOCK_SIZE = re.compile(r'block_size="(\d+)"')

    def __init__(self) -> None:
        self._num_total_tokens: int = 0  # 0 = not yet parsed from cache_config_info

    async def fetch(self, address: str) -> LoadMetrics:
        timeout = aiohttp.ClientTimeout(total=0.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{address}/metrics") as resp:
                text = await resp.text()

        token_usage = 0.0
        num_requests_running = 0
        num_requests_waiting = 0
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if any(line.startswith(m) for m in self._CACHE_METRICS):
                token_usage = float(line.split()[-1])
            elif line.startswith("vllm:num_requests_running"):
                num_requests_running = int(float(line.split()[-1]))
            elif line.startswith("vllm:num_requests_waiting"):
                num_requests_waiting = int(float(line.split()[-1]))
            elif self._num_total_tokens == 0 and line.startswith("vllm:cache_config_info"):
                m_blocks = self._RE_NUM_GPU_BLOCKS.search(line)
                m_bsize = self._RE_BLOCK_SIZE.search(line)
                if m_blocks and m_bsize:
                    self._num_total_tokens = int(m_blocks.group(1)) * int(m_bsize.group(1))
                    logger.info(
                        f"[VLLMLoadBackend] parsed num_total_tokens={self._num_total_tokens} "
                        f"(num_gpu_blocks={m_blocks.group(1)}, block_size={m_bsize.group(1)})"
                    )
        return LoadMetrics(
            token_usage=token_usage,
            num_total_tokens=self._num_total_tokens,
            num_requests_running=num_requests_running,
            num_requests_waiting=num_requests_waiting,
        )


def _make_load_backend(backend: str) -> LoadBackend:
    if backend == "sglang":
        return SGLangLoadBackend()
    if backend == "vllm":
        return VLLMLoadBackend()
    raise ValueError(f"Unknown load backend: {backend!r}. Choose 'sglang' or 'vllm'.")


class _CapacityAwareScheduler:
    """KV-cache-aware load balancer replacing GlobalRequestLoadBalancer.

    Routing policy:
    - Established group (group_id in _affinity): unconditional sticky route,
      preserves prefix-cache warmth regardless of current load.
    - New group: capacity-gated; blocks until a replica has token_usage < threshold.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        capacity_threshold: float = 0.85,
        poll_interval_ms: int = 200,
        load_backend: str = "sglang",
    ):
        self._capacity_threshold = capacity_threshold
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._load_backend: LoadBackend = _make_load_backend(load_backend)
        self._capacity_event = asyncio.Event()
        self._affinity: dict[str, str] = {}
        self._states: dict[str, ReplicaState] = {}
        self._poll_tasks: dict[str, asyncio.Task] = {}

        for server_id, handle in servers.items():
            self._states[server_id] = ReplicaState(
                server_id=server_id,
                handle=handle,
                token_usage=0.0,
                healthy=True,
                last_polled=0.0,
            )

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        for server_id, handle in servers.items():
            self._states[server_id] = ReplicaState(
                server_id=server_id,
                handle=handle,
                token_usage=0.0,
                healthy=True,
                last_polled=0.0,
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.ensure_future(self._poll_loop(server_id))
                    self._poll_tasks[server_id] = task
                # else: no running event loop (e.g. sync test context); poll loop
                # must be started explicitly via _start_poll_loops() once running.
            except RuntimeError:
                # No event loop at all; same fallback as above.
                pass
        logger.info(f"[CapacityAwareScheduler] added {len(servers)} servers")

    async def _start_poll_loops(self) -> None:
        """Start one background poll Task per replica. Call once after __init__ inside Ray actor."""
        for server_id in list(self._states):
            task = asyncio.ensure_future(self._poll_loop(server_id))
            self._poll_tasks[server_id] = task

    @staticmethod
    def _normalize_address(address: str) -> str:
        """Ensure address has an http:// scheme for aiohttp."""
        if address.startswith("http://") or address.startswith("https://"):
            return address
        return f"http://{address}"

    async def _poll_loop(self, server_id: str) -> None:
        address = self._normalize_address(server_id)
        while server_id in self._states:
            try:
                metrics = await self._load_backend.fetch(address)
                state = self._states[server_id]
                state.token_usage = metrics.token_usage
                if metrics.num_total_tokens > 0:
                    state.num_total_tokens = metrics.num_total_tokens
                state.num_requests_running = metrics.num_requests_running
                state.num_requests_waiting = metrics.num_requests_waiting
                # Poll reflects actual server state; reset pending estimate
                state.pending_tokens = 0
                state.healthy = True
                state.last_polled = time.time()
                logger.info(
                    f"[CapacityAwareScheduler] poll ok {server_id} "
                    f"usage={metrics.token_usage:.3f} "
                    f"effective={self._effective_usage(state):.3f} "
                    f"running={metrics.num_requests_running} "
                    f"waiting={metrics.num_requests_waiting}"
                )
                if self._is_available(state):
                    self._capacity_event.set()
            except Exception as exc:
                logger.warning(f"[CapacityAwareScheduler] poll failed for {server_id}: {type(exc).__name__}: {exc}")
                if server_id in self._states:
                    self._states[server_id].healthy = False
            await asyncio.sleep(self._poll_interval_s)

    def _effective_usage(self, state: ReplicaState) -> float:
        """Combine polled token_usage with pending estimate to prevent over-dispatch."""
        if state.num_total_tokens > 0:
            pending_fraction = state.pending_tokens / state.num_total_tokens
        else:
            # Backend doesn't expose total tokens (e.g. vLLM via Prometheus)
            pending_fraction = 0.0
        return min(state.token_usage + pending_fraction, 1.0)

    def _is_available(self, state: ReplicaState) -> bool:
        """A replica is available for new groups when its queue is empty.

        Uses num_requests_waiting as the gate (accurate, polled directly from vLLM).
        Falls back to capacity_threshold when waiting count is unavailable (e.g. SGLang).
        """
        if state.num_requests_waiting > 0:
            return False
        if state.num_requests_running == 0 and state.num_requests_waiting == 0:
            # Both zero could mean metrics not yet populated; fall back to KV-cache check
            return self._effective_usage(state) < self._capacity_threshold
        return True

    async def acquire_server(
        self,
        request_id: str,
        group_id: str | None = None,
        estimated_tokens: int = 0,
    ) -> tuple[str, ray.actor.ActorHandle]:
        key = group_id or request_id

        # Path A: established group — unconditional sticky route
        # Preserves prefix-cache warmth; server handles queuing internally
        if key in self._affinity:
            server_id = self._affinity[key]
            self._states[server_id].pending_tokens += estimated_tokens
            return server_id, self._states[server_id].handle

        # Path B: new group — capacity-gated dispatch
        while True:
            best = self._best_available()
            if best is not None:
                best.pending_tokens += estimated_tokens
                self._affinity[key] = best.server_id
                return best.server_id, best.handle
            self._capacity_event.clear()
            await self._capacity_event.wait()

    def _best_available(self) -> ReplicaState | None:
        available = [s for s in self._states.values() if s.healthy and self._is_available(s)]
        if not available:
            return None
        min_running = min(s.num_requests_running for s in available)
        candidates = [s for s in available if s.num_requests_running - min_running <= 1]
        return random.choice(candidates)

    def release_server(self, server_id: str) -> None:  # noqa: ARG002
        # server_id kept for API compatibility with GlobalRequestLoadBalancer.
        # pending_tokens resets to 0 on the next poll cycle (which reflects actual state).
        self._capacity_event.set()

    def remove_servers(self, server_ids: list[str]) -> None:
        for sid in server_ids:
            task = self._poll_tasks.pop(sid, None)
            if task:
                task.cancel()
            self._states.pop(sid, None)
        logger.info(f"[CapacityAwareScheduler] removed {len(server_ids)} servers")

    def clear_affinity(self) -> None:
        self._affinity.clear()

    def get_status(self) -> dict:
        return {
            "replicas": {
                sid: {
                    "token_usage": s.token_usage,
                    "effective_usage": self._effective_usage(s),
                    "pending_tokens": s.pending_tokens,
                    "num_total_tokens": s.num_total_tokens,
                    "num_requests_running": s.num_requests_running,
                    "num_requests_waiting": s.num_requests_waiting,
                    "healthy": s.healthy,
                    "last_polled": s.last_polled,
                }
                for sid, s in self._states.items()
            },
            "affinity_count": len(self._affinity),
            "capacity_threshold": self._capacity_threshold,
        }


# Expose the raw Python class for unit testing, and the Ray actor class for runtime use.
CapacityAwareScheduler = ray.remote(_CapacityAwareScheduler)
