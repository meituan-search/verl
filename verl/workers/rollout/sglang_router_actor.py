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
"""Ray actor wrapper for the sgl-model-gateway Rust Router."""

import logging
import multiprocessing
import os
import socket
import time
from typing import Optional

import ray
import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _router_subprocess(router_args) -> None:
    from sglang_router.router import Router

    router = Router.from_args(router_args)
    router.start()


@ray.remote(num_cpus=1, num_gpus=0, max_concurrency=4)
class SGLangRouterActor:
    """Ray actor that runs the sgl-model-gateway Rust Router in a subprocess.

    The Router provides cache_aware / round_robin / power_of_two / consistent_hashing
    routing and PD-disaggregated mode.  Workers are registered lazily after sglang
    servers are ready (via add_workers / remove_workers) rather than at Router startup,
    to avoid probe-induced scheduler contention during the initial sleep/wake cycle.

    Args:
        worker_urls: HTTP addresses of sglang replica servers.
            Ignored when pd_disaggregation=True (use prefill_urls/decode_urls instead).
        host: Host the Router binds to. Defaults to the node's actual IP address.
        port: Port the Router listens on. Defaults to an OS-assigned free port.
        policy: Routing policy for non-PD mode.
        cache_threshold: cache_aware — route to replica if prefix match rate exceeds this.
        balance_abs_threshold: cache_aware — switch to load-balance if
            (max_load - min_load) > threshold.
        balance_rel_threshold: cache_aware — switch to load-balance if
            max_load > min_load * threshold.
        request_id_headers: HTTP headers the Router checks for a sticky-session key.
        health_check_interval_secs: How often the Router pings each worker's /health.
        health_failure_threshold: Consecutive failures before marking a worker unhealthy.
        health_success_threshold: Consecutive successes before marking a worker healthy.
        pd_disaggregation: Enable PD-disaggregated routing mode.
        prefill_urls: List of (address, bootstrap_port) tuples for prefill servers.
        decode_urls: HTTP addresses of decode servers.
        prefill_policy: Routing policy for the prefill pool (PD mode only).
        decode_policy: Routing policy for the decode pool (PD mode only).
    """

    def __init__(
        self,
        worker_urls: list[str],
        host: Optional[str] = None,
        port: Optional[int] = None,
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
        from sglang_router.router_args import RouterArgs

        # Resolve host and port inside the actor so they reflect the node where
        # the actor is scheduled, eliminating cross-node port races.
        self._public_host = host if host is not None else ray.util.get_node_ip_address().strip("[]")
        self._bind_host = "0.0.0.0"
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
                _s.bind((self._bind_host, 0))
                self._port = _s.getsockname()[1]
        else:
            self._port = port
        self._address = f"http://{self._public_host}:{self._port}"

        self._router_args = RouterArgs(
            worker_urls=worker_urls if not pd_disaggregation else [],
            host=self._bind_host,
            port=self._port,
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
            disable_health_check=True,
            log_level="warn",
        )
        self._proc: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        """Launch the Rust Router in a subprocess and return immediately."""
        if self._proc is not None and self._proc.is_alive():
            logger.warning("SGLangRouterActor.start() called while Router process is already running")
            return

        print(f"SGLangRouterActor: launching Router subprocess on {self._address}", flush=True)
        self._proc = multiprocessing.Process(
            target=_router_subprocess,
            args=(self._router_args,),
            daemon=True,
            name="sglang-rust-router",
        )
        self._proc.start()
        print(f"SGLangRouterActor: Router subprocess started (pid={self._proc.pid})", flush=True)

    def start_and_wait(self, timeout: float = 120.0) -> str:
        """Start the router, wait until ready, and return its address.

        Combines start() + wait_until_ready() + get_address() in one Ray call to
        avoid multiple round-trips during actor initialisation.  If worker_urls were
        provided at construction time, also waits for all workers to pass
        /health_generate before returning.

        Returns:
            The router's public HTTP address, e.g. 'http://10.0.0.1:30001'.
        """
        self.start()
        self.wait_until_ready(timeout=timeout)
        worker_urls = self._router_args.worker_urls
        if worker_urls:
            self._wait_workers_healthy(worker_urls, timeout=timeout)
        print(f"SGLangRouterActor: start_and_wait complete, address={self._address}", flush=True)
        return self._address

    def wait_until_ready(self, timeout: float = 120.0, poll_interval: float = 1.0) -> bool:
        """Poll GET /health until the Router responds 200 OK or timeout expires."""
        deadline = time.monotonic() + timeout
        # Poll via loopback to avoid firewall rules blocking the actor's external IP.
        local_url = f"http://127.0.0.1:{self._port}/health"
        print(f"SGLangRouterActor: wait_until_ready polling {local_url} (timeout={timeout}s)", flush=True)
        while time.monotonic() < deadline:
            try:
                resp = requests.get(local_url, timeout=(2.0, 2.0))
                if resp.status_code == 200:
                    print(f"SGLangRouterActor: Router ready at {self._address}", flush=True)
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"SGLangRouterActor: Router at {self._address} not ready after {timeout}s")

    def get_address(self) -> str:
        """Return the Router's routable HTTP address, e.g. 'http://10.0.0.1:30001'."""
        return self._address

    def is_alive(self) -> bool:
        """Return True if the Router subprocess is still running."""
        return self._proc is not None and self._proc.is_alive()

    def _wait_workers_healthy(self, worker_urls: list[str], timeout: float = 60.0) -> None:
        """Poll /health_generate on each worker until all respond 200 or timeout expires."""
        deadline = time.monotonic() + timeout
        pending = list(worker_urls)
        while time.monotonic() < deadline and pending:
            still_pending = []
            for url in pending:
                try:
                    r = requests.get(f"{url}/health_generate", timeout=(2.0, 5.0))
                    if r.status_code == 200:
                        continue
                except requests.RequestException:
                    pass
                still_pending.append(url)
            pending = still_pending
            if pending:
                time.sleep(1.0)
        if pending:
            print(f"SGLangRouterActor: WARNING — workers not ready after {timeout}s: {pending}", flush=True)
        else:
            print(f"SGLangRouterActor: all {len(worker_urls)} workers ready for generation", flush=True)

    def remove_workers(self, worker_urls: list[str]) -> None:
        """Remove workers from the Router by URL."""
        admin_url = f"http://127.0.0.1:{self._port}/workers"
        try:
            resp = requests.get(admin_url, timeout=(2.0, 5.0))
            if resp.status_code != 200:
                print(f"SGLangRouterActor: list workers failed: {resp.status_code}", flush=True)
                return
            workers = resp.json()
        except requests.RequestException as e:
            print(f"SGLangRouterActor: list workers error: {e}", flush=True)
            return

        url_set = set(worker_urls)
        worker_list = workers.get("workers", []) if isinstance(workers, dict) else workers
        for w in worker_list:
            if w.get("url") in url_set:
                worker_id = w.get("id")
                try:
                    r = requests.delete(f"http://127.0.0.1:{self._port}/workers/{worker_id}", timeout=(2.0, 5.0))
                    print(f"SGLangRouterActor: removed worker {w['url']} (id={worker_id}): {r.status_code}", flush=True)
                except requests.RequestException as e:
                    print(f"SGLangRouterActor: remove worker {w['url']} error: {e}", flush=True)

    def add_workers(self, worker_urls: list[str], wait_healthy_timeout: float = 60.0) -> None:
        """Register workers with the Router and wait until sglang is ready to serve."""
        admin_url = f"http://127.0.0.1:{self._port}/workers"
        for url in worker_urls:
            try:
                r = requests.post(admin_url, json={"url": url}, timeout=(2.0, 5.0))
                print(f"SGLangRouterActor: added worker {url}: {r.status_code}", flush=True)
            except requests.RequestException as e:
                print(f"SGLangRouterActor: add worker {url} error: {e}", flush=True)
        self._wait_workers_healthy(worker_urls, timeout=wait_healthy_timeout)
