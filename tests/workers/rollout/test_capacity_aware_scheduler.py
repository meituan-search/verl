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
    LoadMetrics,
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
        metrics = await backend.fetch("http://host:8000")
    assert metrics.token_usage == pytest.approx(0.4)
    assert metrics.num_total_tokens == 1000


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
        metrics = await backend.fetch("http://host:9000")
    assert metrics.token_usage == pytest.approx(0.72)
    assert metrics.num_total_tokens == 0  # vLLM Prometheus does not expose raw token count


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


# ---------------------------------------------------------------------------
# Task 4: _poll_loop + _start_poll_loops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_loop_updates_token_usage():
    sched, _ = _make_scheduler(("http://h0:8000",))
    sched._poll_interval_s = 0.01
    call_count = 0

    async def fake_fetch(address):
        nonlocal call_count
        call_count += 1
        return LoadMetrics(token_usage=0.6, num_total_tokens=10000)

    sched._load_backend.fetch = fake_fetch
    task = asyncio.ensure_future(sched._poll_loop("http://h0:8000"))
    await asyncio.sleep(0.08)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert call_count >= 2
    assert sched._states["http://h0:8000"].token_usage == pytest.approx(0.6)
    assert sched._states["http://h0:8000"].healthy is True


@pytest.mark.asyncio
async def test_poll_loop_marks_unhealthy_on_error():
    sched, _ = _make_scheduler(("http://h0:8000",))
    sched._poll_interval_s = 0.01

    async def fail_fetch(address):
        raise Exception("connection refused")

    sched._load_backend.fetch = fail_fetch
    task = asyncio.ensure_future(sched._poll_loop("http://h0:8000"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert sched._states["http://h0:8000"].healthy is False


@pytest.mark.asyncio
async def test_poll_loop_sets_capacity_event_when_below_threshold():
    sched, _ = _make_scheduler(("http://h0:8000",))
    sched._poll_interval_s = 0.01
    sched._capacity_event.clear()

    async def fetch_low(address):
        return LoadMetrics(token_usage=0.3, num_total_tokens=10000)

    sched._load_backend.fetch = fetch_low
    task = asyncio.ensure_future(sched._poll_loop("http://h0:8000"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert sched._capacity_event.is_set()


# ---------------------------------------------------------------------------
# Task 5: acquire_server, _best_available, release_server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_server_new_group_routes_to_least_loaded():
    sched, handles = _make_scheduler(("http://h0:8000", "http://h1:8000"))
    sched._states["http://h0:8000"].token_usage = 0.3
    sched._states["http://h1:8000"].token_usage = 0.6
    sched._capacity_event.set()

    server_id, handle = await sched.acquire_server("req-1", group_id="group-1")
    assert server_id == "http://h0:8000"
    assert sched._affinity["group-1"] == "http://h0:8000"


@pytest.mark.asyncio
async def test_acquire_server_sticky_group_bypasses_capacity():
    sched, handles = _make_scheduler(("http://h0:8000", "http://h1:8000"))
    sched._states["http://h0:8000"].token_usage = 0.95  # over threshold
    sched._states["http://h1:8000"].token_usage = 0.3
    sched._affinity["group-2"] = "http://h0:8000"
    sched._capacity_event.set()

    server_id, handle = await sched.acquire_server("req-2", group_id="group-2")
    assert server_id == "http://h0:8000"  # sticky despite over threshold


@pytest.mark.asyncio
async def test_acquire_server_blocks_when_all_full_then_unblocks():
    sched, handles = _make_scheduler(("http://h0:8000",))
    sched._states["http://h0:8000"].token_usage = 0.95
    sched._capacity_event.clear()

    acquired = []

    async def do_acquire():
        sid, _ = await sched.acquire_server("req-3", group_id="group-3")
        acquired.append(sid)

    task = asyncio.ensure_future(do_acquire())
    await asyncio.sleep(0.02)
    assert len(acquired) == 0  # still blocked

    sched._states["http://h0:8000"].token_usage = 0.5
    sched._capacity_event.set()
    await asyncio.sleep(0.02)

    assert acquired == ["http://h0:8000"]
    task.cancel()


@pytest.mark.asyncio
async def test_acquire_server_no_group_id_falls_back_to_request_id():
    sched, handles = _make_scheduler(("http://h0:8000", "http://h1:8000"))
    sched._states["http://h0:8000"].token_usage = 0.4
    sched._states["http://h1:8000"].token_usage = 0.7
    sched._capacity_event.set()

    server_id, _ = await sched.acquire_server("req-4")
    assert server_id == "http://h0:8000"
    assert sched._affinity.get("req-4") == "http://h0:8000"


def test_release_server_sets_capacity_event():
    sched, _ = _make_scheduler()
    sched._capacity_event.clear()
    sched.release_server("http://h0:8000")
    assert sched._capacity_event.is_set()


def test_best_available_randomizes_within_epsilon():
    sched, _ = _make_scheduler(("http://h0:8000", "http://h1:8000", "http://h2:8000"))
    sched._states["http://h0:8000"].token_usage = 0.30
    sched._states["http://h1:8000"].token_usage = 0.32
    sched._states["http://h2:8000"].token_usage = 0.80  # over threshold

    seen = set()
    for _ in range(40):
        result = sched._best_available()
        seen.add(result.server_id)
    assert seen == {"http://h0:8000", "http://h1:8000"}


# ---------------------------------------------------------------------------
# Task 6: LLMServerClient group_id forwarding
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_server_client_passes_group_id_to_acquire():
    """LLMServerClient.generate() must forward group_id to acquire_server.remote()."""
    from verl.workers.rollout.llm_server import LLMServerClient
    from verl.workers.rollout.replica import TokenOutput

    mock_output = TokenOutput(token_ids=[1, 2, 3], log_probs=None, stop_reason="stop", extra_fields={})

    mock_lb = MagicMock()
    mock_handle = MagicMock()
    mock_lb.acquire_server.remote = AsyncMock(return_value=("http://h0:8000", mock_handle))
    mock_lb.release_server.remote = MagicMock()
    mock_handle.generate.remote = AsyncMock(return_value=mock_output)

    cfg = MagicMock()
    client = LLMServerClient(config=cfg, load_balancer_handle=mock_lb)
    await client.generate("req-1", group_id="group-A", prompt_ids=[10, 20], sampling_params={})

    mock_lb.acquire_server.remote.assert_called_once()
    call_kwargs = mock_lb.acquire_server.remote.call_args.kwargs
    assert call_kwargs.get("group_id") == "group-A"
