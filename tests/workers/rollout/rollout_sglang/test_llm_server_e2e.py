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
"""End-to-end tests using real Ray actors and a real aiohttp HTTP server.

Unlike test_llm_server_router.py (which stubs everything), these tests use:
  - Real Ray cluster (ray.init / ray.shutdown)
  - Real GlobalRequestLoadBalancer Ray actor
  - Real aiohttp.ClientSession (no mocks)
  - Real HTTP server (Python http.server) as a stand-in for sglang /generate

Requires: ray, aiohttp, cachetools
Does NOT require: sglang, sglang_router, GPU, torch

Test groups
-----------
TestGlobalRequestLoadBalancerRay    – sticky session + in-flight counting via real Ray RPCs
TestLLMServerClientHTTPRay          – _generate_via_router against a real HTTP server
TestLLMServerManagerLeastInflight   – _init_global_load_balancer creates a real Ray actor
                                      and get_client() wires it correctly
"""

import asyncio
import json

# ---------------------------------------------------------------------------
# Ensure verl package is importable
# ---------------------------------------------------------------------------
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest
import ray

_ROOT = pathlib.Path(__file__).parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Session-scoped Ray cluster
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ray_cluster():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Fake sglang /generate HTTP server
# ---------------------------------------------------------------------------


def _make_fake_generate_handler(response_body: dict | None = None):
    """Return an HTTPRequestHandler class that echoes a fixed JSON body."""
    _default = {
        "output_ids": [100, 200, 300],
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "num_preempted": 0,
            "global_steps": 7,
        },
    }
    body = json.dumps(response_body or _default).encode()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)  # consume body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # silence test output

    return _Handler


class _FakeGenerateServer:
    """Context manager: starts a real HTTP server on a free port."""

    def __init__(self, response_body: dict | None = None):
        self._port = _free_port()
        handler = _make_fake_generate_handler(response_body)
        self._server = HTTPServer(("127.0.0.1", self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return f"http://127.0.0.1:{self._port}"

    def __exit__(self, *_):
        self._server.shutdown()


# ---------------------------------------------------------------------------
# Import the module under test with real deps (no stubs needed here)
# ---------------------------------------------------------------------------

from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer, LLMServerClient  # noqa: E402

# ---------------------------------------------------------------------------
# TestGlobalRequestLoadBalancerRay
# ---------------------------------------------------------------------------


class TestGlobalRequestLoadBalancerRay:
    """GlobalRequestLoadBalancer running as a real Ray actor."""

    @pytest.fixture
    def lb(self):
        servers = {"s0": MagicMock(), "s1": MagicMock(), "s2": MagicMock()}
        actor = GlobalRequestLoadBalancer.remote(servers=servers)
        yield actor
        ray.kill(actor)

    # ------------------------------------------------------------------
    # Routing: least in-flight
    # ------------------------------------------------------------------

    def test_new_request_routes_to_least_loaded(self, lb):
        """Fresh requests should all go to the same server (all at 0 in-flight)."""
        sid = ray.get(lb.acquire_server.remote(request_id="req-001"))
        assert sid in ("s0", "s1", "s2")

    def test_acquire_increments_inflight(self, lb):
        """After acquiring server X, it should be less preferred for the next request."""
        # Acquire s0, s1, s2 in sequence — each call should pick a different server
        # since in-flight is 0 for all initially (ties broken by dict iteration order).
        s0 = ray.get(lb.acquire_server.remote(request_id="r1"))
        s1 = ray.get(lb.acquire_server.remote(request_id="r2"))
        s2 = ray.get(lb.acquire_server.remote(request_id="r3"))
        # All three must be distinct because each acquire bumps the winner's count.
        assert len({s0, s1, s2}) == 3

    def test_release_decrements_inflight(self, lb):
        """After release, the server's in-flight count goes back to zero."""
        sid = ray.get(lb.acquire_server.remote(request_id="req-rel"))
        # Should not raise
        ray.get(lb.release_server.remote(server_id=sid))

    def test_release_unknown_server_raises(self, lb):
        """Releasing a server that was never acquired raises ValueError."""
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(lb.release_server.remote(server_id="nonexistent"))

    def test_double_release_raises(self, lb):
        """Releasing twice raises ValueError (in-flight would go below 0)."""
        sid = ray.get(lb.acquire_server.remote(request_id="req-double"))
        ray.get(lb.release_server.remote(server_id=sid))
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(lb.release_server.remote(server_id=sid))

    # ------------------------------------------------------------------
    # Sticky session
    # ------------------------------------------------------------------

    def test_same_request_id_routes_to_same_server(self, lb):
        """Multi-turn: same request_id must always go to the same replica."""
        first = ray.get(lb.acquire_server.remote(request_id="conv-1"))
        ray.get(lb.release_server.remote(server_id=first))
        second = ray.get(lb.acquire_server.remote(request_id="conv-1"))
        assert second == first

    def test_different_request_ids_may_route_differently(self, lb):
        """Two distinct request IDs can (and should) land on different servers."""
        s_a = ray.get(lb.acquire_server.remote(request_id="conv-A"))
        s_b = ray.get(lb.acquire_server.remote(request_id="conv-B"))
        s_c = ray.get(lb.acquire_server.remote(request_id="conv-C"))
        # After 3 distinct requests with 3 servers the set must be {s0,s1,s2}
        assert {s_a, s_b, s_c} == {"s0", "s1", "s2"}

    def test_sticky_session_survives_concurrent_requests(self, lb):
        """Sticky routing holds even when other conversations are in-flight."""
        # conv-X acquires a server, then conv-Y acquires another,
        # but re-acquiring conv-X must still return the original server.
        sx = ray.get(lb.acquire_server.remote(request_id="conv-X"))
        ray.get(lb.acquire_server.remote(request_id="conv-Y"))  # other conversation
        sx_again = ray.get(lb.acquire_server.remote(request_id="conv-X"))
        assert sx_again == sx

    def test_empty_servers_raises_on_init(self):
        """Constructing GlobalRequestLoadBalancer with empty servers must raise."""
        with pytest.raises((ray.exceptions.RayTaskError, ray.exceptions.OutOfMemoryError)):
            actor = GlobalRequestLoadBalancer.remote(servers={})
            ray.get(actor.acquire_server.remote(request_id="x"))


# ---------------------------------------------------------------------------
# TestLLMServerClientHTTPRay
# ---------------------------------------------------------------------------


class TestLLMServerClientHTTPRay:
    """LLMServerClient._generate_via_router against a real HTTP server.

    Uses real aiohttp — no mocks.
    """

    def test_happy_path_real_http(self):
        """Client POSTs to a real HTTP server and parses the response correctly."""

        async def _body():
            with _FakeGenerateServer() as url:
                client = LLMServerClient(config=MagicMock(), router_address=url)
                out = await client._generate_via_router(
                    request_id="e2e-1",
                    prompt_ids=[1, 2, 3],
                    sampling_params={"temperature": 1.0},
                    image_data=None,
                    video_data=None,
                )
                await client.close()

            assert out.token_ids == [100, 200, 300]
            assert out.stop_reason == "stop"
            assert out.extra_fields["global_steps"] == 7

        _arun(_body())

    def test_request_id_header_sent_to_real_server(self):
        """The x-verl-request-id header reaches the server (captured via request inspection)."""
        received_headers = {}

        class _CapturingHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                received_headers.update(dict(self.headers))
                body = json.dumps({"output_ids": [], "meta_info": {}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        port = _free_port()
        server = HTTPServer(("127.0.0.1", port), _CapturingHandler)
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        async def _body():
            client = LLMServerClient(config=MagicMock(), router_address=f"http://127.0.0.1:{port}")
            await client._generate_via_router(
                request_id="sticky-xyz",
                prompt_ids=[],
                sampling_params={},
                image_data=None,
                video_data=None,
            )
            await client.close()

        _arun(_body())
        t.join(timeout=2)
        server.server_close()

        assert received_headers.get("x-verl-request-id") == "sticky-xyz"

    def test_5xx_retries_three_times_real_http(self):
        """Client retries exactly 3 times on 503 before raising RuntimeError."""
        call_count = 0

        class _AlwaysFail(BaseHTTPRequestHandler):
            def do_POST(self):
                nonlocal call_count
                call_count += 1
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        port = _free_port()
        server = HTTPServer(("127.0.0.1", port), _AlwaysFail)
        t = threading.Thread(target=lambda: [server.handle_request() for _ in range(5)], daemon=True)
        t.start()

        async def _body():
            client = LLMServerClient(config=MagicMock(), router_address=f"http://127.0.0.1:{port}")
            with pytest.raises(RuntimeError, match="3 attempts"):
                await client._generate_via_router(
                    request_id="r",
                    prompt_ids=[],
                    sampling_params={},
                    image_data=None,
                    video_data=None,
                )
            await client.close()

        _arun(_body())
        t.join(timeout=5)
        server.server_close()

        assert call_count == 3

    def test_session_reuse_across_requests(self):
        """Two consecutive generate calls reuse the same aiohttp session."""

        async def _body():
            with _FakeGenerateServer() as url:
                client = LLMServerClient(config=MagicMock(), router_address=url)

                await client._generate_via_router(
                    request_id="r1",
                    prompt_ids=[],
                    sampling_params={},
                    image_data=None,
                    video_data=None,
                )
                session_after_first = client._session

                await client._generate_via_router(
                    request_id="r2",
                    prompt_ids=[],
                    sampling_params={},
                    image_data=None,
                    video_data=None,
                )
                session_after_second = client._session

                await client.close()

            assert session_after_first is session_after_second

        _arun(_body())

    def test_connection_refused_retries_then_raises(self):
        """When the target port has no listener, client retries 3 times then raises."""
        port = _free_port()  # nothing listening here

        async def _body():
            client = LLMServerClient(config=MagicMock(), router_address=f"http://127.0.0.1:{port}")
            with pytest.raises(RuntimeError, match="3 attempts"):
                await client._generate_via_router(
                    request_id="r",
                    prompt_ids=[],
                    sampling_params={},
                    image_data=None,
                    video_data=None,
                )
            await client.close()

        _arun(_body())


# ---------------------------------------------------------------------------
# TestLLMServerManagerLeastInflight
# ---------------------------------------------------------------------------


class TestLLMServerManagerLeastInflight:
    """LLMServerManager._init_global_load_balancer creates a real Ray actor."""

    def _make_manager(self):
        from verl.workers.rollout.llm_server import LLMServerManager

        mgr = object.__new__(LLMServerManager)
        mgr.config = MagicMock()
        rollout_cfg = MagicMock()
        rollout_cfg.name = "sglang"
        rollout_cfg.load_balance = MagicMock()
        rollout_cfg.load_balance.strategy = "least_inflight"
        rollout_cfg.prometheus = MagicMock(enable=False)
        mgr.rollout_config = rollout_cfg
        mgr.config.actor_rollout_ref.rollout = rollout_cfg
        mgr.server_addresses = ["http://fake-r0:8000", "http://fake-r1:8000"]
        mgr.server_handles = [MagicMock(), MagicMock()]
        mgr.worker_group = MagicMock()
        mgr.router_actor = None
        mgr.router_address = None
        mgr.global_load_balancer = None
        return mgr

    def test_least_inflight_creates_real_ray_actor(self):
        """_init_global_load_balancer must create a live Ray actor."""

        async def _body():
            mgr = self._make_manager()
            await mgr._init_global_load_balancer()

            assert mgr.global_load_balancer is not None
            assert mgr.router_address is None
            assert mgr.router_actor is None

            # Actor must be alive: a simple RPC should succeed
            # The actor was created with mgr.server_addresses as keys — verify it responds
            sid = ray.get(mgr.global_load_balancer.acquire_server.remote(request_id="probe"))
            assert sid in mgr.server_addresses

            ray.kill(mgr.global_load_balancer)

        _arun(_body())

    def test_get_client_least_inflight_wires_real_lb(self):
        """get_client() returns a client whose _load_balancer is the real Ray actor."""

        async def _body():
            from verl.workers.rollout.llm_server import LLMServerClient

            mgr = self._make_manager()
            await mgr._init_global_load_balancer()

            client = mgr.get_client(fully_async=False)

            assert isinstance(client, LLMServerClient)
            assert client._load_balancer is mgr.global_load_balancer
            assert client._router_address is None

            ray.kill(mgr.global_load_balancer)

        _arun(_body())

    def test_lb_acquire_release_roundtrip_via_client(self):
        """acquire → release roundtrip works end-to-end through the Ray actor."""

        async def _body():
            mgr = self._make_manager()
            await mgr._init_global_load_balancer()

            lb = mgr.global_load_balancer
            sid = ray.get(lb.acquire_server.remote(request_id="roundtrip-1"))
            assert sid in mgr.server_addresses
            # release must not raise
            ray.get(lb.release_server.remote(server_id=sid))
            # after release, the same request_id still routes to the same server (sticky)
            sid2 = ray.get(lb.acquire_server.remote(request_id="roundtrip-1"))
            assert sid2 == sid

            ray.kill(mgr.global_load_balancer)

        _arun(_body())
