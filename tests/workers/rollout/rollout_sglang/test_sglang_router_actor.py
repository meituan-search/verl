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
"""Unit tests for SGLangRouterActor.

All tests run without a real Rust Router binary or sglang_router_rs extension —
the Rust objects are replaced by a minimal pure-Python fake that exercises the
same interface. This keeps the tests fast and runnable in any environment.

Test groups
-----------
TestSGLangRouterActorInit      – constructor argument handling and RouterArgs construction
TestSGLangRouterActorStart     – daemon-thread lifecycle
TestSGLangRouterActorReady     – wait_until_ready() polling and timeout
TestSGLangRouterActorAccessors – get_address() and is_alive()
TestSGLangRouterActorIdempotent – double-start guard
TestSGLangRouterActorPDMode    – PD-disaggregation parameter path
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: fake sglang_router objects injected via sys.modules patches
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Minimal stand-in for sglang_router.sglang_router_rs.Router (Rust object).

    start() blocks until self._stop is set, simulating the real blocking call.
    """

    def __init__(self):
        self._stop = threading.Event()
        self.start_called = False

    def start(self):
        self.start_called = True
        self._stop.wait()  # block until stop() is called

    def stop(self):
        self._stop.set()


class _FakeRouterClass:
    """Stand-in for the Router Python wrapper returned by Router.from_args()."""

    def __init__(self):
        self._inner = _FakeRouter()

    @classmethod
    def from_args(cls, args):
        instance = cls()
        instance._captured_args = args
        return instance

    def start(self):
        self._inner.start()

    def stop(self):
        self._inner.stop()


class _FakeRouterArgs:
    """Stand-in for sglang_router.router_args.RouterArgs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Fixture: patch sglang_router imports before the module is imported
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_sglang_router(monkeypatch):
    """Replace sglang_router.* with fakes so no Rust binary is needed."""
    import sys
    import types

    # Build a fake package tree
    pkg = types.ModuleType("sglang_router")
    router_mod = types.ModuleType("sglang_router.router")
    args_mod = types.ModuleType("sglang_router.router_args")

    router_mod.Router = _FakeRouterClass
    args_mod.RouterArgs = _FakeRouterArgs

    pkg.router = router_mod
    pkg.router_args = args_mod

    monkeypatch.setitem(sys.modules, "sglang_router", pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.router", router_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.router_args", args_mod)

    # Also patch ray so we can import the actor class without a Ray cluster
    ray_mock = MagicMock()
    ray_mock.remote = lambda *a, **kw: (lambda cls: cls)  # identity decorator
    monkeypatch.setitem(sys.modules, "ray", ray_mock)

    yield


@pytest.fixture
def ActorClass():
    """Load SGLangRouterActor directly from its source file.

    We bypass the verl package __init__ (which pulls in tensordict / torch / ray)
    by using importlib.util.spec_from_file_location.  The module is loaded fresh
    each time so that the monkeypatched sys.modules entries are picked up.
    """
    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).parents[4] / "verl" / "workers" / "rollout" / "sglang_router_actor.py"
    spec = importlib.util.spec_from_file_location("sglang_router_actor_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SGLangRouterActor


@pytest.fixture
def default_kwargs():
    return dict(
        worker_urls=["http://localhost:8001", "http://localhost:8002"],
        host="127.0.0.1",
        port=30001,
    )


# ---------------------------------------------------------------------------
# TestSGLangRouterActorInit
# ---------------------------------------------------------------------------


class TestSGLangRouterActorInit:
    def test_address_format(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        assert actor._address == "http://127.0.0.1:30001"
        assert actor._host == "127.0.0.1"
        assert actor._port == 30001

    def test_thread_initially_none(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        assert actor._thread is None

    def test_default_request_id_headers(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        args = actor._router._captured_args
        assert args.request_id_headers == ["x-verl-request-id"]

    def test_custom_request_id_headers(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs, request_id_headers=["x-my-id"])
        args = actor._router._captured_args
        assert args.request_id_headers == ["x-my-id"]

    def test_worker_urls_passed_through(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        args = actor._router._captured_args
        assert args.worker_urls == default_kwargs["worker_urls"]

    def test_policy_passed_through(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs, policy="round_robin")
        args = actor._router._captured_args
        assert args.policy == "round_robin"

    def test_cache_aware_thresholds(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs, cache_threshold=0.5, balance_abs_threshold=32, balance_rel_threshold=2.0)
        args = actor._router._captured_args
        assert args.cache_threshold == 0.5
        assert args.balance_abs_threshold == 32
        assert args.balance_rel_threshold == 2.0

    def test_health_check_params(self, ActorClass, default_kwargs):
        actor = ActorClass(
            **default_kwargs,
            health_check_interval_secs=30,
            health_failure_threshold=5,
            health_success_threshold=3,
        )
        args = actor._router._captured_args
        assert args.health_check_interval_secs == 30
        assert args.health_failure_threshold == 5
        assert args.health_success_threshold == 3


# ---------------------------------------------------------------------------
# TestSGLangRouterActorPDMode
# ---------------------------------------------------------------------------


class TestSGLangRouterActorPDMode:
    def test_pd_mode_worker_urls_cleared(self, ActorClass):
        actor = ActorClass(
            worker_urls=["http://irrelevant:8001"],
            host="0.0.0.0",
            port=30001,
            pd_disaggregation=True,
            prefill_urls=[("http://prefill:8001", 9001)],
            decode_urls=["http://decode:8002"],
        )
        args = actor._router._captured_args
        # When pd_disaggregation=True, worker_urls must be empty
        assert args.worker_urls == []

    def test_pd_mode_prefill_decode_urls(self, ActorClass):
        actor = ActorClass(
            worker_urls=[],
            host="0.0.0.0",
            port=30001,
            pd_disaggregation=True,
            prefill_urls=[("http://prefill:8001", 9001)],
            decode_urls=["http://decode:8002", "http://decode:8003"],
        )
        args = actor._router._captured_args
        assert args.pd_disaggregation is True
        assert args.prefill_urls == [("http://prefill:8001", 9001)]
        assert args.decode_urls == ["http://decode:8002", "http://decode:8003"]

    def test_pd_mode_separate_policies(self, ActorClass):
        actor = ActorClass(
            worker_urls=[],
            host="0.0.0.0",
            port=30001,
            pd_disaggregation=True,
            prefill_urls=[("http://prefill:8001", 9001)],
            decode_urls=["http://decode:8002"],
            prefill_policy="cache_aware",
            decode_policy="power_of_two",
        )
        args = actor._router._captured_args
        assert args.prefill_policy == "cache_aware"
        assert args.decode_policy == "power_of_two"

    def test_pd_mode_empty_prefill_urls_defaults(self, ActorClass):
        actor = ActorClass(
            worker_urls=[],
            host="0.0.0.0",
            port=30001,
            pd_disaggregation=True,
        )
        args = actor._router._captured_args
        assert args.prefill_urls == []
        assert args.decode_urls == []


# ---------------------------------------------------------------------------
# TestSGLangRouterActorStart
# ---------------------------------------------------------------------------


class TestSGLangRouterActorStart:
    def test_start_spawns_daemon_thread(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        assert actor._thread is not None
        assert actor._thread.is_alive()
        assert actor._thread.daemon is True
        assert actor._thread.name == "sglang-rust-router"
        # cleanup
        actor._router._inner.stop()
        actor._thread.join(timeout=2)

    def test_start_calls_router_start(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        # Give the thread a moment to call router.start()
        deadline = time.monotonic() + 2.0
        while not actor._router._inner.start_called and time.monotonic() < deadline:
            time.sleep(0.05)
        assert actor._router._inner.start_called
        actor._router._inner.stop()
        actor._thread.join(timeout=2)

    def test_start_sets_thread_name(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        assert actor._thread.name == "sglang-rust-router"
        actor._router._inner.stop()
        actor._thread.join(timeout=2)

    def test_thread_exits_after_router_stops(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        actor._router._inner.stop()
        actor._thread.join(timeout=3)
        assert not actor._thread.is_alive()


# ---------------------------------------------------------------------------
# TestSGLangRouterActorIdempotent
# ---------------------------------------------------------------------------


class TestSGLangRouterActorIdempotent:
    def test_double_start_does_not_spawn_second_thread(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        first_thread = actor._thread

        actor.start()  # second call — should be a no-op
        assert actor._thread is first_thread  # same thread object

        actor._router._inner.stop()
        first_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# TestSGLangRouterActorReady
# ---------------------------------------------------------------------------


class TestSGLangRouterActorReady:
    def test_wait_until_ready_success(self, ActorClass, default_kwargs):
        """When /health returns 200, wait_until_ready returns True."""
        actor = ActorClass(**default_kwargs)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            result = actor.wait_until_ready(timeout=5.0, poll_interval=0.05)

        assert result is True

    def test_wait_until_ready_timeout(self, ActorClass, default_kwargs):
        """When /health never returns 200, TimeoutError is raised."""
        actor = ActorClass(**default_kwargs)

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(TimeoutError, match="not ready after"):
                actor.wait_until_ready(timeout=0.2, poll_interval=0.05)

    def test_wait_until_ready_connection_errors_then_success(self, ActorClass, default_kwargs):
        """Connection errors are swallowed; eventual 200 succeeds."""
        import requests as req_lib

        actor = ActorClass(**default_kwargs)

        ok_resp = MagicMock()
        ok_resp.status_code = 200

        call_count = 0

        def side_effect(url, timeout):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise req_lib.ConnectionError("refused")
            return ok_resp

        with patch("requests.get", side_effect=side_effect):
            result = actor.wait_until_ready(timeout=5.0, poll_interval=0.05)

        assert result is True
        assert call_count >= 3

    def test_wait_until_ready_non_200_then_200(self, ActorClass, default_kwargs):
        """Non-200 responses are retried until 200 or timeout."""
        actor = ActorClass(**default_kwargs)

        responses = iter(
            [
                MagicMock(status_code=503),
                MagicMock(status_code=503),
                MagicMock(status_code=200),
            ]
        )

        with patch("requests.get", side_effect=lambda *a, **kw: next(responses)):
            result = actor.wait_until_ready(timeout=5.0, poll_interval=0.05)

        assert result is True

    def test_wait_until_ready_polls_correct_url(self, ActorClass):
        actor = ActorClass(worker_urls=[], host="10.0.0.5", port=12345)

        ok_resp = MagicMock()
        ok_resp.status_code = 200

        with patch("requests.get", return_value=ok_resp) as mock_get:
            actor.wait_until_ready(timeout=5.0, poll_interval=0.05)

        mock_get.assert_called_with("http://10.0.0.5:12345/health", timeout=2.0)

    def test_wait_until_ready_request_timeout_retried(self, ActorClass, default_kwargs):
        """requests.Timeout (a subclass of RequestException) is also retried."""
        import requests as req_lib

        actor = ActorClass(**default_kwargs)
        ok_resp = MagicMock(status_code=200)
        responses = [req_lib.Timeout("timed out"), ok_resp]

        with patch("requests.get", side_effect=responses):
            result = actor.wait_until_ready(timeout=5.0, poll_interval=0.05)

        assert result is True


# ---------------------------------------------------------------------------
# TestSGLangRouterActorAccessors
# ---------------------------------------------------------------------------


class TestSGLangRouterActorAccessors:
    def test_get_address_format(self, ActorClass):
        actor = ActorClass(worker_urls=[], host="192.168.1.1", port=9999)
        assert actor.get_address() == "http://192.168.1.1:9999"

    def test_get_address_localhost(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        assert actor.get_address() == "http://127.0.0.1:30001"

    def test_is_alive_before_start(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        assert actor.is_alive() is False

    def test_is_alive_after_start(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        assert actor.is_alive() is True
        actor._router._inner.stop()
        actor._thread.join(timeout=2)

    def test_is_alive_after_router_stops(self, ActorClass, default_kwargs):
        actor = ActorClass(**default_kwargs)
        actor.start()
        actor._router._inner.stop()
        actor._thread.join(timeout=3)
        assert actor.is_alive() is False


# ---------------------------------------------------------------------------
# Integration-style test: real HTTP server as mock worker
# ---------------------------------------------------------------------------


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that responds 200 to GET /health."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # silence output during tests


class TestWaitUntilReadyRealHTTP:
    """Use a real local HTTP server to verify end-to-end polling behaviour."""

    def test_polls_real_health_endpoint(self, ActorClass):
        """wait_until_ready succeeds when a real server responds 200 on /health."""
        server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
        port = server.server_address[1]

        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        actor = ActorClass(worker_urls=[], host="127.0.0.1", port=port)
        result = actor.wait_until_ready(timeout=5.0, poll_interval=0.1)

        assert result is True
        server.server_close()

    def test_timeout_with_no_server(self, ActorClass):
        """wait_until_ready raises TimeoutError when nothing is listening."""
        actor = ActorClass(worker_urls=[], host="127.0.0.1", port=19999)
        with pytest.raises(TimeoutError):
            actor.wait_until_ready(timeout=0.3, poll_interval=0.1)
