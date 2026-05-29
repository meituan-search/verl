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
                return d["num_used_tokens"] / d["num_total_tokens"]


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
