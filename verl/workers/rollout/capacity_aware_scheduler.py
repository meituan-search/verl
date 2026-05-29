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
import random
import time
from dataclasses import dataclass
from typing import Protocol

import aiohttp
import ray

logger = logging.getLogger(__name__)


@dataclass
class LoadMetrics:
    token_usage: float       # 0.0–1.0, KV cache utilization
    num_total_tokens: int    # total KV cache capacity in tokens; 0 = unknown


@dataclass
class ReplicaState:
    server_id: str                 # also the HTTP base URL, e.g. "http://host:8000"
    handle: ray.actor.ActorHandle
    token_usage: float             # 0.0–1.0, last polled KV cache utilization
    healthy: bool
    last_polled: float             # Unix timestamp
    num_total_tokens: int = 1      # KV cache capacity; updated each poll cycle
    pending_tokens: int = 0        # tokens dispatched since last poll, not yet reflected in token_usage


class LoadBackend(Protocol):
    async def fetch(self, address: str) -> LoadMetrics:
        """Return current load metrics for the replica at address."""
        ...


class SGLangLoadBackend:
    """Fetches load metrics from sglang /v1/loads endpoint."""

    async def fetch(self, address: str) -> LoadMetrics:
        timeout = aiohttp.ClientTimeout(total=0.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{address}/v1/loads") as resp:
                data = await resp.json()
                d = data[0]
                total = d["num_total_tokens"]
                usage = d["num_used_tokens"] / total if total > 0 else 0.0
                return LoadMetrics(token_usage=usage, num_total_tokens=total)


class VLLMLoadBackend:
    """Fetches load metrics from vLLM Prometheus /metrics endpoint."""

    async def fetch(self, address: str) -> LoadMetrics:
        timeout = aiohttp.ClientTimeout(total=0.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{address}/metrics") as resp:
                text = await resp.text()
        for line in text.splitlines():
            if line.startswith("vllm:gpu_cache_usage_perc") and not line.startswith("#"):
                # vLLM Prometheus does not expose raw token counts; pending estimation disabled
                return LoadMetrics(token_usage=float(line.split()[-1]), num_total_tokens=0)
        return LoadMetrics(token_usage=0.0, num_total_tokens=0)


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

    async def _poll_loop(self, server_id: str) -> None:
        while server_id in self._states:
            try:
                metrics = await self._load_backend.fetch(server_id)
                state = self._states[server_id]
                state.token_usage = metrics.token_usage
                if metrics.num_total_tokens > 0:
                    state.num_total_tokens = metrics.num_total_tokens
                # Poll reflects actual server state; reset pending estimate
                state.pending_tokens = 0
                state.healthy = True
                state.last_polled = time.time()
                if self._effective_usage(state) < self._capacity_threshold:
                    self._capacity_event.set()
            except Exception as exc:
                logger.warning(f"[CapacityAwareScheduler] poll failed for {server_id}: {exc}")
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
        available = [
            s for s in self._states.values()
            if s.healthy and self._effective_usage(s) < self._capacity_threshold
        ]
        if not available:
            return None
        min_usage = min(self._effective_usage(s) for s in available)
        candidates = [s for s in available if self._effective_usage(s) - min_usage < 0.05]
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
