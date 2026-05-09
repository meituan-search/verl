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
"""Ray actor wrapper for the sgl-model-gateway Rust Router.

The Rust Router's Router.start() is a blocking call that runs a high-performance
HTTP reverse-proxy with pluggable load-balancing policies (cache_aware, round_robin,
power_of_two, etc.) and built-in health-check / circuit-breaker logic.

This module wraps it in a Ray actor so that its lifecycle is managed by Ray
alongside the rest of the verl rollout infrastructure.

Lifecycle:
    1. SGLangRouterActor.__init__: construct the Router object (fast, no I/O)
    2. SGLangRouterActor.start(): spawn a daemon thread and return immediately
    3. SGLangRouterActor.wait_until_ready(timeout): poll /health until 200 OK
    4. Actor death: daemon thread exits with the actor process
"""

import logging
import threading
import time
from typing import Optional

import ray
import requests

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0)
class SGLangRouterActor:
    """Ray actor that runs the sgl-model-gateway Rust Router in a daemon thread.

    The Router provides:
    - cache_aware routing (radix-tree prefix matching across replicas)
    - round_robin / power_of_two / consistent_hashing policies
    - periodic /health checks with circuit-breaker per worker
    - PD-disaggregated mode (separate prefill/decode policies)

    Args:
        worker_urls: HTTP addresses of sglang replica servers.
            Ignored when pd_disaggregation=True (use prefill_urls/decode_urls instead).
        host: Host the Router binds to.
        port: Port the Router listens on.
        policy: Routing policy for non-PD mode.
            Options: cache_aware, round_robin, consistent_hashing, prefix_hash.
        cache_threshold: cache_aware — route to replica if prefix match rate exceeds this.
        balance_abs_threshold: cache_aware — switch to load-balance if
            (max_load - min_load) > threshold.
        balance_rel_threshold: cache_aware — switch to load-balance if
            max_load > min_load * threshold.
        request_id_headers: HTTP headers the Router checks for a sticky-session key.
            Defaults to ["x-verl-request-id"].
        health_check_interval_secs: How often the Router pings each worker's /health.
        health_failure_threshold: Consecutive failures before marking a worker unhealthy.
        health_success_threshold: Consecutive successes before marking a worker healthy again.
        pd_disaggregation: Enable PD-disaggregated routing mode.
        prefill_urls: List of (address, bootstrap_port) tuples for prefill servers.
        decode_urls: HTTP addresses of decode servers.
        prefill_policy: Routing policy for the prefill pool (PD mode only).
        decode_policy: Routing policy for the decode pool (PD mode only).
            power_of_two is recommended for decode servers.
    """

    def __init__(
        self,
        worker_urls: list[str],
        host: str,
        port: int,
        policy: str = "cache_aware",
        cache_threshold: float = 0.3,
        balance_abs_threshold: int = 64,
        balance_rel_threshold: float = 1.5,
        request_id_headers: Optional[list[str]] = None,
        health_check_interval_secs: int = 60,
        health_failure_threshold: int = 3,
        health_success_threshold: int = 2,
        pd_disaggregation: bool = False,
        prefill_urls: Optional[list[tuple[str, int]]] = None,
        decode_urls: Optional[list[str]] = None,
        prefill_policy: Optional[str] = None,
        decode_policy: Optional[str] = None,
    ):
        from sglang_router.router import Router
        from sglang_router.router_args import RouterArgs

        self._host = host
        self._port = port
        self._address = f"http://{host}:{port}"

        args = RouterArgs(
            worker_urls=worker_urls if not pd_disaggregation else [],
            host=host,
            port=port,
            policy=policy,
            cache_threshold=cache_threshold,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            request_id_headers=request_id_headers or ["x-verl-request-id"],
            health_check_interval_secs=health_check_interval_secs,
            health_failure_threshold=health_failure_threshold,
            health_success_threshold=health_success_threshold,
            pd_disaggregation=pd_disaggregation,
            prefill_urls=prefill_urls or [],
            decode_urls=decode_urls or [],
            prefill_policy=prefill_policy,
            decode_policy=decode_policy,
        )
        self._router = Router.from_args(args)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the Rust Router in a daemon thread and return immediately.

        The thread is a daemon so it is automatically killed when the Ray actor
        process exits, leaving no zombie ports.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("SGLangRouterActor.start() called while Router thread is already running")
            return

        self._thread = threading.Thread(
            target=self._router.start,
            daemon=True,
            name="sglang-rust-router",
        )
        self._thread.start()
        logger.info("SGLangRouterActor: Router thread started on %s", self._address)

    def wait_until_ready(self, timeout: float = 120.0, poll_interval: float = 1.0) -> bool:
        """Poll GET /health until the Router responds 200 OK or timeout expires.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between each probe.

        Returns:
            True when the Router is ready.

        Raises:
            TimeoutError: If the Router is not ready within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        url = f"{self._address}/health"
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("SGLangRouterActor: Router ready at %s", self._address)
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"SGLangRouterActor: Router at {self._address} not ready after {timeout}s")

    def get_address(self) -> str:
        """Return the Router's base HTTP address, e.g. 'http://0.0.0.0:30001'."""
        return self._address

    def is_alive(self) -> bool:
        """Return True if the Router daemon thread is still running."""
        return self._thread is not None and self._thread.is_alive()
