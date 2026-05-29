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


# ---------------------------------------------------------------------------
# Tests for CapacityAwareScheduler skeleton
# ---------------------------------------------------------------------------
from verl.workers.rollout.capacity_aware_scheduler import _CapacityAwareScheduler


def _make_scheduler(addresses=("http://h0:8000", "http://h1:8000")):
    handles = {addr: MagicMock() for addr in addresses}
    # Use the raw (non-Ray-decorated) class so object.__new__ works without a Ray cluster
    sched = object.__new__(_CapacityAwareScheduler)
    _CapacityAwareScheduler.__init__(sched, servers=handles, capacity_threshold=0.85, poll_interval_ms=200, load_backend="sglang")
    return sched, handles


def test_scheduler_init_creates_replica_states():
    sched, handles = _make_scheduler()
    assert len(sched._states) == 2
    for addr in handles:
        assert sched._states[addr].server_id == addr
        assert sched._states[addr].token_usage == 0.0
        assert sched._states[addr].healthy is True


def test_add_servers():
    sched, _ = _make_scheduler(("http://h0:8000",))
    new_handle = MagicMock()
    sched.add_servers({"http://h2:8000": new_handle})
    assert "http://h2:8000" in sched._states
    assert sched._states["http://h2:8000"].handle is new_handle


def test_remove_servers_cleans_state():
    sched, _ = _make_scheduler()
    mock_task = MagicMock()
    mock_task.cancel = MagicMock()
    sched._poll_tasks["http://h0:8000"] = mock_task
    sched.remove_servers(["http://h0:8000"])
    assert "http://h0:8000" not in sched._states
    mock_task.cancel.assert_called_once()


def test_clear_affinity():
    sched, _ = _make_scheduler()
    sched._affinity["group_1"] = "http://h0:8000"
    sched.clear_affinity()
    assert len(sched._affinity) == 0


def test_get_status():
    sched, _ = _make_scheduler()
    sched._states["http://h0:8000"].token_usage = 0.4
    status = sched.get_status()
    assert "replicas" in status
    assert status["replicas"]["http://h0:8000"]["token_usage"] == 0.4
    assert "affinity_count" in status
    assert "capacity_threshold" in status
