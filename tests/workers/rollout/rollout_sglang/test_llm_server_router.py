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
"""Unit tests for the sglang_router path in LLMServerClient and LLMServerManager.

Covers:
  TestParseTokenOutput       – _parse_token_output() JSON → TokenOutput
  TestLLMServerClientRouterInit – LLMServerClient init (router_address vs legacy modes)
  TestGenerateViaRouter      – _generate_via_router() happy path, retries, error handling
  TestGenerateViaRouterLogprobs – prompt_logprobs translation
  TestSessionManagement      – aiohttp session lazy-init, reuse, close
  TestLLMServerManagerInitLB – _init_global_load_balancer() routing decision
  TestLLMServerManagerGetClient – get_client() returns correct client class/mode
  TestLLMServerManagerInitRouter – _init_sglang_router() wires up the Router actor

All tests run without Ray, torch, or tensordict by loading only the files that
are strictly needed via importlib.util.spec_from_file_location.
"""

import asyncio
import importlib.util
import pathlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _arun(coro):
    """Run a coroutine synchronously — replaces pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Bootstrap: load the modules under test without triggering verl/__init__
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parents[4]


def _load_file(dotted_name: str, rel: str):
    path = _ROOT / rel
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_stub(dotted_name: str):
    """Insert an empty stub module if the real one is not yet importable."""
    if dotted_name not in sys.modules:
        parts = dotted_name.split(".")
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            if key not in sys.modules:
                sys.modules[key] = types.ModuleType(key)


# Minimal stubs for heavy deps that llm_server.py imports at the top level.
for _mod in [
    "ray",
    "torch",
    "cachetools",
    "omegaconf",
    "verl",
    "verl.single_controller",
    "verl.single_controller.ray",
    "verl.single_controller.ray.base",
    "verl.utils",
    "verl.utils.ray_utils",
    "verl.utils.rollout_trace",
    "verl.utils.tokenizer",
    "verl.utils.net_utils",
    "verl.workers",
    "verl.workers.rollout",
    "verl.workers.rollout.replica",
    "verl.workers.rollout.utils",
    "verl.workers.rollout.sglang_router_actor",
]:
    _ensure_stub(_mod)

# ray stubs
_ray = sys.modules["ray"]
_ray.remote = lambda *a, **kw: (lambda cls: cls)
_ray.actor = MagicMock()
_ray.actor.ActorHandle = object
_ray.util = MagicMock()

# torch stub
_torch = sys.modules["torch"]
_torch.tensor = lambda x: x  # return as-is for tests

# aiohttp stub with all attributes that llm_server.py references at class-definition time.
_aiohttp_stub = types.ModuleType("aiohttp")


class _FakeClientResponseError(Exception):
    def __init__(self, *a, status=500, **kw):
        super().__init__()
        self.status = status
        self.request_info = kw.get("request_info", MagicMock())
        self.history = kw.get("history", ())


class _FakeClientConnectorError(Exception):
    def __init__(self, *a, **kw):
        super().__init__()


_aiohttp_stub.ClientSession = MagicMock
_aiohttp_stub.ClientTimeout = MagicMock
_aiohttp_stub.TCPConnector = MagicMock
_aiohttp_stub.ClientResponseError = _FakeClientResponseError
_aiohttp_stub.ClientConnectorError = _FakeClientConnectorError
sys.modules["aiohttp"] = _aiohttp_stub

# cachetools stub
_ct = sys.modules["cachetools"]
_ct.LRUCache = dict  # simple dict as stand-in (enough for tests that don't hit LB)

# omegaconf stub — temporarily replace DictConfig so llm_server.py's type
# annotation `DictConfig` resolves to a plain dict during module load.
# We restore the real DictConfig immediately after _load_file() below so that
# other test files in the same pytest process are not affected.
_omegaconf = sys.modules["omegaconf"]
_omegaconf_real_DictConfig = getattr(_omegaconf, "DictConfig", None)
_omegaconf.DictConfig = dict

# verl.utils.ray_utils stub
_ray_utils = sys.modules["verl.utils.ray_utils"]
_ray_utils.auto_await = lambda fn: fn

# verl.utils.rollout_trace stub — identity decorator
_rt = sys.modules["verl.utils.rollout_trace"]
_rt.rollout_trace_op = lambda fn: fn

# verl.utils.tokenizer stub
_tok = sys.modules["verl.utils.tokenizer"]
_tok.normalize_token_ids = lambda ids: ids

# verl.workers.rollout.replica stubs
_replica_mod = sys.modules["verl.workers.rollout.replica"]


class _TokenOutput:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        if not hasattr(self, "extra_fields"):
            self.extra_fields = {}
        if not hasattr(self, "log_probs"):
            self.log_probs = None
        if not hasattr(self, "routed_experts"):
            self.routed_experts = None
        if not hasattr(self, "num_preempted"):
            self.num_preempted = None
        if not hasattr(self, "stop_reason"):
            self.stop_reason = None


_replica_mod.TokenOutput = _TokenOutput
_replica_mod.RolloutReplica = object
_replica_mod.get_rollout_replica_class = MagicMock(return_value=MagicMock)

# verl.workers.rollout.utils stub
_rutils = sys.modules["verl.workers.rollout.utils"]
_rutils.update_prometheus_config = MagicMock()

# verl.utils.net_utils stub
_net_utils = sys.modules["verl.utils.net_utils"]
_net_utils.get_free_port = MagicMock(return_value=30001)

# verl.workers.rollout.sglang_router_actor stub
_router_actor_stub = sys.modules["verl.workers.rollout.sglang_router_actor"]
_router_actor_stub.SGLangRouterActor = MagicMock()

# single_controller stubs
sys.modules["verl.single_controller.ray.base"].RayResourcePool = object
sys.modules["verl.single_controller.ray.base"].RayWorkerGroup = object

# Now load the module under test
_llm_server_mod = _load_file(
    "verl.workers.rollout.llm_server",
    "verl/workers/rollout/llm_server.py",
)

# Restore the real omegaconf.DictConfig now that the module is loaded.
# This prevents polluting other test files that run in the same pytest process.
if _omegaconf_real_DictConfig is not None:
    _omegaconf.DictConfig = _omegaconf_real_DictConfig

# Pull out symbols
_parse_token_output = _llm_server_mod._parse_token_output
LLMServerClient = _llm_server_mod.LLMServerClient
FullyLLMServerClient = _llm_server_mod.FullyLLMServerClient
LLMServerManager = _llm_server_mod.LLMServerManager
GlobalRequestLoadBalancer = _llm_server_mod.GlobalRequestLoadBalancer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(strategy="least_inflight", rollout_name="sglang"):
    """Build a minimal config mock."""
    router_cfg = MagicMock()
    router_cfg.policy = "cache_aware"
    router_cfg.cache_threshold = 0.3
    router_cfg.balance_abs_threshold = 64
    router_cfg.balance_rel_threshold = 1.5
    router_cfg.health_check_interval_secs = 60

    lb_cfg = MagicMock()
    lb_cfg.strategy = strategy
    lb_cfg.router = router_cfg

    rollout_cfg = MagicMock()
    rollout_cfg.name = rollout_name
    rollout_cfg.load_balance = lb_cfg
    rollout_cfg.prometheus = MagicMock(enable=False)
    rollout_cfg.disable_log_stats = True
    rollout_cfg.disaggregation = MagicMock(enabled=False)
    rollout_cfg.nnodes = 1
    rollout_cfg.n_gpus_per_node = 8
    rollout_cfg.tensor_model_parallel_size = 1
    rollout_cfg.data_parallel_size = 1
    rollout_cfg.pipeline_model_parallel_size = 1

    cfg = MagicMock()
    cfg.actor_rollout_ref.rollout = rollout_cfg
    cfg.actor_rollout_ref.model = MagicMock()
    return cfg


# ---------------------------------------------------------------------------
# TestParseTokenOutput
# ---------------------------------------------------------------------------


class TestParseTokenOutput:
    def test_basic_token_ids(self):
        data = {"output_ids": [10, 20, 30], "meta_info": {}}
        out = _parse_token_output(data)
        assert out.token_ids == [10, 20, 30]

    def test_empty_output_ids(self):
        data = {"output_ids": [], "meta_info": {}}
        out = _parse_token_output(data)
        assert out.token_ids == []

    def test_missing_output_ids(self):
        out = _parse_token_output({})
        assert out.token_ids == []

    def test_stop_reason_from_finish_reason(self):
        data = {
            "output_ids": [1],
            "meta_info": {"finish_reason": {"type": "stop"}},
        }
        out = _parse_token_output(data)
        assert out.stop_reason == "stop"

    def test_stop_reason_none_when_absent(self):
        out = _parse_token_output({"output_ids": [1], "meta_info": {}})
        assert out.stop_reason is None

    def test_log_probs_extracted_when_lengths_match(self):
        data = {
            "output_ids": [1, 2],
            "meta_info": {
                "output_token_logprobs": [[-0.5, 1, 0], [-1.2, 2, 0]],
            },
        }
        out = _parse_token_output(data)
        assert out.log_probs == pytest.approx([-0.5, -1.2])

    def test_log_probs_none_when_lengths_mismatch(self):
        data = {
            "output_ids": [1, 2, 3],
            "meta_info": {"output_token_logprobs": [[-0.5, 1, 0]]},
        }
        out = _parse_token_output(data)
        assert out.log_probs is None

    def test_log_probs_none_when_absent(self):
        out = _parse_token_output({"output_ids": [1], "meta_info": {}})
        assert out.log_probs is None

    def test_num_preempted_default_zero(self):
        out = _parse_token_output({"output_ids": [], "meta_info": {}})
        assert out.num_preempted == 0

    def test_num_preempted_extracted(self):
        data = {"output_ids": [], "meta_info": {"num_preempted": 3}}
        out = _parse_token_output(data)
        assert out.num_preempted == 3

    def test_global_steps_in_extra_fields(self):
        data = {"output_ids": [], "meta_info": {"global_steps": 42}}
        out = _parse_token_output(data)
        assert out.extra_fields["global_steps"] == 42

    def test_global_steps_absent_means_empty_extra_fields(self):
        out = _parse_token_output({"output_ids": [], "meta_info": {}})
        assert "global_steps" not in out.extra_fields

    def test_routed_experts_none_when_absent(self):
        out = _parse_token_output({"output_ids": [], "meta_info": {}})
        assert out.routed_experts is None

    def test_routed_experts_converted(self):
        data = {"output_ids": [1], "meta_info": {"routed_experts": [[0, 1], [2, 3]]}}
        out = _parse_token_output(data)
        # torch.tensor is stubbed to return as-is
        assert out.routed_experts == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# TestLLMServerClientRouterInit
# ---------------------------------------------------------------------------


class TestLLMServerClientRouterInit:
    def test_router_address_stored(self):
        cfg = MagicMock()
        client = LLMServerClient(config=cfg, router_address="http://127.0.0.1:9000")
        assert client._router_address == "http://127.0.0.1:9000"
        assert client._load_balancer is None
        assert client._server_id_to_handle == {}

    def test_legacy_path_stored(self):
        cfg = MagicMock()
        lb = MagicMock()
        servers = {"addr1": MagicMock()}
        client = LLMServerClient(config=cfg, servers=servers, load_balancer_handle=lb)
        assert client._router_address is None
        assert client._load_balancer is lb
        assert client._server_id_to_handle is servers

    def test_session_initially_none(self):
        client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
        assert client._session is None


# ---------------------------------------------------------------------------
# TestGenerateViaRouter
# ---------------------------------------------------------------------------


class TestGenerateViaRouter:
    """Tests for LLMServerClient._generate_via_router()."""

    def _make_client(self, router_address="http://router:9000"):
        return LLMServerClient(config=MagicMock(), router_address=router_address)

    def _mock_aiohttp_response(self, json_data: dict, status: int = 200):
        resp = AsyncMock()
        resp.status = status
        resp.json = AsyncMock(return_value=json_data)
        resp.raise_for_status = MagicMock()  # no-op on 200
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        return resp, cm

    def test_happy_path_returns_token_output(self):
        async def _body():
            client = self._make_client()
            response_data = {
                "output_ids": [100, 200],
                "meta_info": {"finish_reason": {"type": "stop"}, "global_steps": 5},
            }
            resp, cm = self._mock_aiohttp_response(response_data)

            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            out = await client._generate_via_router(
                request_id="req-1",
                prompt_ids=[1, 2, 3],
                sampling_params={"temperature": 1.0},
                image_data=None,
                video_data=None,
            )

            assert out.token_ids == [100, 200]
            assert out.stop_reason == "stop"
            assert out.extra_fields["global_steps"] == 5

        _arun(_body())

    def test_request_id_header_injected(self):
        async def _body():
            client = self._make_client()
            response_data = {"output_ids": [1], "meta_info": {}}
            resp, cm = self._mock_aiohttp_response(response_data)

            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            await client._generate_via_router(
                request_id="sticky-id-xyz",
                prompt_ids=[1],
                sampling_params={},
                image_data=None,
                video_data=None,
            )

            call_kwargs = mock_session.post.call_args[1]
            assert call_kwargs["headers"]["x-verl-request-id"] == "sticky-id-xyz"

        _arun(_body())

    def test_url_constructed_from_router_address(self):
        async def _body():
            client = self._make_client("http://myrouter:7777")
            response_data = {"output_ids": [], "meta_info": {}}
            resp, cm = self._mock_aiohttp_response(response_data)

            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            await client._generate_via_router(
                request_id="r", prompt_ids=[], sampling_params={}, image_data=None, video_data=None
            )

            url = mock_session.post.call_args[0][0]
            assert url == "http://myrouter:7777/generate"

        _arun(_body())

    def test_4xx_raises_immediately_no_retry(self):
        async def _body():
            import aiohttp as _aiohttp_real

            client = self._make_client()

            call_count = 0

            async def _raise_400(*a, **kw):
                nonlocal call_count
                call_count += 1
                raise _aiohttp_real.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=400,
                )

            mock_session = AsyncMock()
            mock_session.closed = False
            # post() must return a context manager whose __aenter__ raises
            cm = AsyncMock()
            cm.__aenter__ = _raise_400
            cm.__aexit__ = AsyncMock(return_value=False)
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            with pytest.raises(_aiohttp_real.ClientResponseError):
                await client._generate_via_router(
                    request_id="r", prompt_ids=[], sampling_params={}, image_data=None, video_data=None
                )

            assert call_count == 1  # no retry on 4xx

        _arun(_body())

    def test_5xx_retries_then_raises_runtime_error(self):
        async def _body():
            import aiohttp as _aiohttp_real

            client = self._make_client()

            async def _raise_503(*a, **kw):
                raise _aiohttp_real.ClientResponseError(request_info=MagicMock(), history=(), status=503)

            cm = AsyncMock()
            cm.__aenter__ = _raise_503
            cm.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(RuntimeError, match="3 attempts"):
                    await client._generate_via_router(
                        request_id="r", prompt_ids=[], sampling_params={}, image_data=None, video_data=None
                    )

            assert mock_session.post.call_count == 3

        _arun(_body())

    def test_connector_error_retries_then_raises(self):
        async def _body():
            import aiohttp as _aiohttp_real

            client = self._make_client()

            async def _raise_conn(*a, **kw):
                raise _aiohttp_real.ClientConnectorError(connection_key=MagicMock(), os_error=OSError("refused"))

            cm = AsyncMock()
            cm.__aenter__ = _raise_conn
            cm.__aexit__ = AsyncMock(return_value=False)
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(RuntimeError, match="3 attempts"):
                    await client._generate_via_router(
                        request_id="r", prompt_ids=[], sampling_params={}, image_data=None, video_data=None
                    )

            assert mock_session.post.call_count == 3

        _arun(_body())


# ---------------------------------------------------------------------------
# TestGenerateViaRouterLogprobs
# ---------------------------------------------------------------------------


class TestGenerateViaRouterLogprobs:
    def _make_client(self):
        return LLMServerClient(config=MagicMock(), router_address="http://router:9000")

    def test_prompt_logprobs_translated_to_sglang_params(self):
        async def _body():
            client = self._make_client()
            response_data = {"output_ids": [1], "meta_info": {}}
            resp = AsyncMock()
            resp.json = AsyncMock(return_value=response_data)
            resp.raise_for_status = MagicMock()
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            # pass prompt_logprobs=3 (vLLM style)
            await client._generate_via_router(
                request_id="r",
                prompt_ids=[1, 2],
                sampling_params={"prompt_logprobs": 3},
                image_data=None,
                video_data=None,
            )

            payload = mock_session.post.call_args[1]["json"]
            assert payload["return_logprob"] is True
            assert payload["logprob_start_len"] == 0
            assert payload["top_logprobs_num"] == 3
            # original key must be consumed
            assert "prompt_logprobs" not in payload.get("sampling_params", {})

        _arun(_body())

    def test_logprobs_zero_sets_return_logprob(self):
        async def _body():
            client = self._make_client()
            response_data = {"output_ids": [1], "meta_info": {}}
            resp = AsyncMock()
            resp.json = AsyncMock(return_value=response_data)
            resp.raise_for_status = MagicMock()
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.post = MagicMock(return_value=cm)
            client._session = mock_session

            await client._generate_via_router(
                request_id="r",
                prompt_ids=[1],
                sampling_params={"logprobs": True},
                image_data=None,
                video_data=None,
            )

            payload = mock_session.post.call_args[1]["json"]
            assert payload["return_logprob"] is True
            assert "top_logprobs_num" not in payload

        _arun(_body())


# ---------------------------------------------------------------------------
# TestSessionManagement
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_session_created_lazily(self):
        async def _body():
            import aiohttp as _aiohttp_real

            client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
            assert client._session is None

            mock_session = MagicMock()
            mock_session.closed = False

            with patch.object(_aiohttp_real, "TCPConnector", return_value=MagicMock()):
                with patch.object(_aiohttp_real, "ClientSession", return_value=mock_session):
                    sess = await client._get_or_create_session()

            assert sess is mock_session
            assert client._session is mock_session

        _arun(_body())

    def test_session_reused_on_second_call(self):
        async def _body():
            import aiohttp as _aiohttp_real

            client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
            mock_session = MagicMock()
            mock_session.closed = False
            client._session = mock_session

            with patch.object(_aiohttp_real, "ClientSession") as mock_cls:
                sess = await client._get_or_create_session()

            mock_cls.assert_not_called()  # no new session created
            assert sess is mock_session

        _arun(_body())

    def test_closed_session_is_recreated(self):
        async def _body():
            import aiohttp as _aiohttp_real

            client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
            old_session = MagicMock()
            old_session.closed = True
            client._session = old_session

            new_session = MagicMock()
            new_session.closed = False

            with patch.object(_aiohttp_real, "TCPConnector", return_value=MagicMock()):
                with patch.object(_aiohttp_real, "ClientSession", return_value=new_session):
                    sess = await client._get_or_create_session()

            assert sess is new_session

        _arun(_body())

    def test_close_closes_session(self):
        async def _body():
            client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
            mock_session = AsyncMock()
            mock_session.closed = False
            client._session = mock_session

            await client.close()
            mock_session.close.assert_awaited_once()

        _arun(_body())

    def test_close_noop_when_session_none(self):
        async def _body():
            client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
            # Should not raise
            await client.close()

        _arun(_body())

    def test_close_noop_when_already_closed(self):
        async def _body():
            client = LLMServerClient(config=MagicMock(), router_address="http://x:1")
            mock_session = AsyncMock()
            mock_session.closed = True
            client._session = mock_session

            await client.close()
            mock_session.close.assert_not_awaited()

        _arun(_body())


# ---------------------------------------------------------------------------
# TestLLMServerManagerInitLB
# ---------------------------------------------------------------------------


class TestLLMServerManagerInitLB:
    """Test _init_global_load_balancer() routing decision."""

    def _make_manager(self, strategy="least_inflight", rollout_name="sglang"):
        mgr = object.__new__(LLMServerManager)
        mgr.config = _mock_config(strategy=strategy, rollout_name=rollout_name)
        mgr.rollout_config = mgr.config.actor_rollout_ref.rollout
        mgr.server_addresses = ["http://r0:8000", "http://r1:8000"]
        mgr.server_handles = [MagicMock(), MagicMock()]
        mgr.worker_group = MagicMock()  # HYBRID mode (non-None)
        return mgr

    def test_least_inflight_creates_global_lb(self):
        async def _body():
            mgr = self._make_manager(strategy="least_inflight")
            mock_lb_instance = MagicMock()
            mock_lb_cls = MagicMock()
            mock_lb_cls.remote = MagicMock(return_value=mock_lb_instance)

            with patch.object(_llm_server_mod, "GlobalRequestLoadBalancer", mock_lb_cls):
                await mgr._init_global_load_balancer()

            mock_lb_cls.remote.assert_called_once()
            assert mgr.router_address is None
            assert mgr.router_actor is None

        _arun(_body())

    def test_sglang_router_strategy_with_non_sglang_engine_falls_back(self):
        async def _body():
            """strategy=sglang_router + rollout.name!=sglang → falls back to least_inflight."""
            mgr = self._make_manager(strategy="sglang_router", rollout_name="vllm")
            mock_lb_cls = MagicMock()
            mock_lb_cls.remote = MagicMock(return_value=MagicMock())

            with patch.object(_llm_server_mod, "GlobalRequestLoadBalancer", mock_lb_cls):
                await mgr._init_global_load_balancer()

            mock_lb_cls.remote.assert_called_once()
            assert mgr.router_address is None

        _arun(_body())

    def test_missing_load_balance_config_defaults_to_least_inflight(self):
        async def _body():
            mgr = self._make_manager()
            del mgr.rollout_config.load_balance

            mock_lb_cls = MagicMock()
            mock_lb_cls.remote = MagicMock(return_value=MagicMock())

            with patch.object(_llm_server_mod, "GlobalRequestLoadBalancer", mock_lb_cls):
                await mgr._init_global_load_balancer()

            mock_lb_cls.remote.assert_called_once()

        _arun(_body())

    def test_sglang_router_strategy_calls_init_sglang_router(self):
        async def _body():
            mgr = self._make_manager(strategy="sglang_router", rollout_name="sglang")

            with patch.object(mgr, "_init_sglang_router", new=AsyncMock()) as mock_init:
                await mgr._init_global_load_balancer()

            mock_init.assert_awaited_once()

        _arun(_body())


# ---------------------------------------------------------------------------
# TestLLMServerManagerGetClient
# ---------------------------------------------------------------------------


class TestLLMServerManagerGetClient:
    def _make_manager(self, router_address=None):
        mgr = object.__new__(LLMServerManager)
        mgr.config = _mock_config()
        mgr.server_addresses = ["http://r0:8000"]
        mgr.server_handles = [MagicMock()]
        mgr.global_load_balancer = MagicMock() if router_address is None else None
        mgr.router_address = router_address
        return mgr

    def test_least_inflight_returns_llm_server_client(self):
        mgr = self._make_manager(router_address=None)
        client = mgr.get_client(fully_async=False)
        assert isinstance(client, LLMServerClient)
        assert client._router_address is None
        assert client._load_balancer is mgr.global_load_balancer

    def test_least_inflight_fully_async_returns_fully_client(self):
        mgr = self._make_manager(router_address=None)
        client = mgr.get_client(fully_async=True)
        assert isinstance(client, FullyLLMServerClient)
        assert client._router_address is None

    def test_router_mode_returns_client_with_router_address(self):
        mgr = self._make_manager(router_address="http://router:9000")
        client = mgr.get_client(fully_async=False)
        assert isinstance(client, LLMServerClient)
        assert client._router_address == "http://router:9000"

    def test_router_mode_fully_async_returns_fully_client(self):
        mgr = self._make_manager(router_address="http://router:9000")
        client = mgr.get_client(fully_async=True)
        assert isinstance(client, FullyLLMServerClient)
        assert client._router_address == "http://router:9000"

    def test_router_mode_client_has_no_load_balancer(self):
        mgr = self._make_manager(router_address="http://router:9000")
        client = mgr.get_client()
        assert client._load_balancer is None

    def test_multiple_get_client_calls_return_independent_objects(self):
        mgr = self._make_manager(router_address="http://router:9000")
        c1 = mgr.get_client()
        c2 = mgr.get_client()
        assert c1 is not c2
        assert c1._router_address == c2._router_address


# ---------------------------------------------------------------------------
# TestLLMServerManagerInitRouter
# ---------------------------------------------------------------------------


class TestLLMServerManagerInitRouter:
    def _make_manager(self):
        mgr = object.__new__(LLMServerManager)
        mgr.config = _mock_config(strategy="sglang_router")
        mgr.rollout_config = mgr.config.actor_rollout_ref.rollout
        mgr.server_addresses = ["http://r0:8000", "http://r1:8000"]
        mgr.server_handles = [MagicMock(), MagicMock()]
        mgr.worker_group = MagicMock()  # HYBRID (non-None → skip node-affinity)
        return mgr

    def test_router_actor_started_and_address_stored(self):
        async def _body():
            mgr = self._make_manager()
            lb_cfg = mgr.rollout_config.load_balance

            # Ray actors expose .method.remote() which returns an awaitable.
            # Simulate: actor.start.remote() -> awaitable, etc.
            mock_actor = MagicMock()
            mock_actor.start.remote = AsyncMock(return_value=None)
            mock_actor.wait_until_ready.remote = AsyncMock(return_value=True)
            mock_actor.get_address.remote = AsyncMock(return_value="http://0.0.0.0:30001")

            mock_actor_cls = MagicMock()
            mock_actor_cls.options.return_value.remote.return_value = mock_actor

            # Patch at the module level where _init_sglang_router does its import
            _router_actor_stub.SGLangRouterActor = mock_actor_cls
            _net_utils.get_free_port = MagicMock(return_value=30001)
            await mgr._init_sglang_router(lb_cfg)

            assert mgr.router_address == "http://0.0.0.0:30001"
            assert mgr.router_actor is mock_actor
            assert mgr.global_load_balancer is None

        _arun(_body())

    def test_worker_urls_are_server_addresses(self):
        async def _body():
            """The Router must receive all replica addresses for initial registration."""
            mgr = self._make_manager()
            lb_cfg = mgr.rollout_config.load_balance

            captured_kwargs = {}

            mock_actor = MagicMock()
            mock_actor.start.remote = AsyncMock(return_value=None)
            mock_actor.wait_until_ready.remote = AsyncMock(return_value=True)
            mock_actor.get_address.remote = AsyncMock(return_value="http://0.0.0.0:30001")

            class _CapturingCls:
                def options(self, **kw):
                    return self

                def remote(self, **kw):
                    captured_kwargs.update(kw)
                    return mock_actor

            _router_actor_stub.SGLangRouterActor = _CapturingCls()
            _net_utils.get_free_port = MagicMock(return_value=30001)
            await mgr._init_sglang_router(lb_cfg)

            assert captured_kwargs["worker_urls"] == ["http://r0:8000", "http://r1:8000"]
            assert captured_kwargs["request_id_headers"] == ["x-verl-request-id"]

        _arun(_body())
