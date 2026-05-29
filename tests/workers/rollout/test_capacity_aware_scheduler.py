# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""GPU-free unit tests for LoadBalanceConfig dataclass."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from verl.workers.config.rollout import LoadBalanceConfig


def test_load_balance_config_defaults():
    cfg = LoadBalanceConfig()
    assert cfg.capacity_threshold == 0.85
    assert cfg.poll_interval_ms == 200
    assert cfg.backend == "sglang"


def test_load_balance_config_from_yaml():
    yaml = """
    capacity_threshold: 0.90
    poll_interval_ms: 100
    backend: vllm
    """
    raw = OmegaConf.create(yaml)
    cfg = LoadBalanceConfig(**raw)
    assert cfg.capacity_threshold == 0.90
    assert cfg.poll_interval_ms == 100
    assert cfg.backend == "vllm"


# ---------------------------------------------------------------------------
# Tests for ReplicaState and LoadBackend implementations
# ---------------------------------------------------------------------------
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verl.workers.rollout.capacity_aware_scheduler import (
    ReplicaState,
    SGLangLoadBackend,
    VLLMLoadBackend,
)


def test_replica_state_fields():
    handle = MagicMock()
    state = ReplicaState(
        server_id="http://host:8000",
        handle=handle,
        token_usage=0.5,
        healthy=True,
        last_polled=time.time(),
    )
    assert state.server_id == "http://host:8000"
    assert state.token_usage == 0.5
    assert state.healthy is True


@pytest.mark.asyncio
async def test_sglang_load_backend_parses_response():
    backend = SGLangLoadBackend()
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=[{"num_used_tokens": 400, "num_total_tokens": 1000}])
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("verl.workers.rollout.capacity_aware_scheduler.aiohttp.ClientSession", return_value=mock_session):
        usage = await backend.fetch("http://host:8000")
    assert usage == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_vllm_load_backend_parses_prometheus():
    backend = VLLMLoadBackend()
    prometheus_text = (
        "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage\n"
        "# TYPE vllm:gpu_cache_usage_perc gauge\n"
        'vllm:gpu_cache_usage_perc{model_name="m"} 0.72\n'
    )
    mock_resp = AsyncMock()
    mock_resp.text = AsyncMock(return_value=prometheus_text)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("verl.workers.rollout.capacity_aware_scheduler.aiohttp.ClientSession", return_value=mock_session):
        usage = await backend.fetch("http://host:9000")
    assert usage == pytest.approx(0.72)
