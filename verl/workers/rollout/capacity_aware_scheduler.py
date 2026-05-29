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
class ReplicaState:
    server_id: str          # also the HTTP base URL, e.g. "http://host:8000"
    handle: ray.actor.ActorHandle
    token_usage: float      # 0.0–1.0, from /v1/loads
    healthy: bool
    last_polled: float      # Unix timestamp


class LoadBackend(Protocol):
    async def fetch(self, address: str) -> float:
        """Return token_usage in [0.0, 1.0]."""
        ...


class SGLangLoadBackend:
    """Fetches token_usage from sglang /v1/loads endpoint."""

    async def fetch(self, address: str) -> float:
        timeout = aiohttp.ClientTimeout(total=0.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{address}/v1/loads") as resp:
                data = await resp.json()
                d = data[0]
                total = d["num_total_tokens"]
                return d["num_used_tokens"] / total if total > 0 else 0.0


class VLLMLoadBackend:
    """Fetches token_usage from vLLM Prometheus /metrics endpoint."""

    async def fetch(self, address: str) -> float:
        timeout = aiohttp.ClientTimeout(total=0.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{address}/metrics") as resp:
                text = await resp.text()
        for line in text.splitlines():
            if line.startswith("vllm:gpu_cache_usage_perc") and not line.startswith("#"):
                return float(line.split()[-1])
        return 0.0


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
        logger.info(f"[CapacityAwareScheduler] added {len(servers)} servers")

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
