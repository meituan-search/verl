# Task 01：SGLangRouterActor — Rust Router 的 Ray 封装

> 新增文件：`verl/workers/rollout/sglang_router_actor.py`
> 依赖：无（最先实现）

---

## 背景

sglang Rust Router（`sgl-model-gateway`）的 Python 入口 `Router.start()` 是一个**阻塞调用**——它在内部启动一个 Rust async HTTP server 并一直运行，直到进程退出。verl 的整个 rollout 基础设施运行在 Ray 上，因此需要把 Router 包装成 Ray actor，使其生命周期由 Ray 托管。

---

## 目标

1. 提供 `SGLangRouterActor`：在 Ray actor 内以 daemon thread 运行 Rust Router
2. 提供 `wait_until_ready()`：轮询 `/health` 直到 Router 可接受请求
3. 提供资源配置：Router 只需 CPU，不占 GPU
4. 提供部署约束接口：STANDALONE 模式下可将 Router 调度到 rollout 节点

---

## 实现

### 文件：`verl/workers/rollout/sglang_router_actor.py`

```python
import logging
import socket
import threading
import time
from typing import Optional

import ray
import requests

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0)
class SGLangRouterActor:
    """Ray actor wrapping the sgl-model-gateway Rust Router.

    The Router's .start() is a blocking call; we run it in a daemon thread so
    the Ray actor event loop stays responsive to health probes and shutdown.

    Lifecycle:
        1. __init__: build Router object (fast, no I/O)
        2. start():  launch daemon thread; returns immediately
        3. wait_until_ready(timeout): poll /health until 200 or timeout
        4. Actor death: daemon thread exits with the actor process
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
        # PD mode
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
        """Launch the Rust Router in a daemon thread. Returns immediately."""
        self._thread = threading.Thread(
            target=self._router.start,
            daemon=True,
            name="sglang-rust-router",
        )
        self._thread.start()
        logger.info(f"SGLangRouterActor: Router thread started on {self._address}")

    def wait_until_ready(self, timeout: float = 120.0, poll_interval: float = 1.0) -> bool:
        """Poll /health until Router is ready or timeout expires.

        Returns True if ready, raises TimeoutError otherwise.
        """
        deadline = time.monotonic() + timeout
        url = f"{self._address}/health"
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info(f"SGLangRouterActor: Router ready at {self._address}")
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(
            f"SGLangRouterActor: Router at {self._address} not ready after {timeout}s"
        )

    def get_address(self) -> str:
        """Return the Router's HTTP address (e.g. 'http://0.0.0.0:30000')."""
        return self._address

    def is_alive(self) -> bool:
        """Return True if the Router thread is still running."""
        return self._thread is not None and self._thread.is_alive()
```

---

## 调用方式（在 `LLMServerManager` 中）

```python
from verl.workers.rollout.sglang_router_actor import SGLangRouterActor
from verl.utils.net_utils import get_free_port

router_port = get_free_port()

# 在 rollout 节点（STANDALONE）或本地（HYBRID/COLOCATED）启动
actor_options = {"num_cpus": 1}
if standalone_mode:
    # 约束到与 replica-0 相同的节点，减少跨节点跳数
    actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
        node_id=rollout_node_id, soft=True
    )

router_actor = SGLangRouterActor.options(**actor_options).remote(
    worker_urls=self.server_addresses,   # 全部 replica HTTP 地址
    host="0.0.0.0",
    port=router_port,
    policy=lb_config.policy,            # 来自 rollout.yaml
    request_id_headers=["x-verl-request-id"],
)
await router_actor.start.remote()
await router_actor.wait_until_ready.remote(timeout=120.0)
self.router_address = await router_actor.get_address.remote()
self.router_actor = router_actor
```

---

## 关键设计决策

### 为什么用 daemon thread 而非 subprocess？

Rust Router 是一个 Python binding（`sglang_router_rs`），`Router.start()` 最终调用 `_Router.start()`，这是 Rust 编写的 Pyo3 binding，在调用期间持有 GIL 直到 Rust server 退出。使用 `daemon=True` 线程确保当 Ray actor 进程退出时 Rust 线程也随之终止，不会有僵尸进程。

使用 subprocess（`multiprocessing.Process`）也可以，但需要额外的 IPC 来传递端口号和健康状态，复杂度更高。

### 为什么 `wait_until_ready()` 不在 `start()` 内部？

Ray remote call 有超时限制。将 wait 逻辑分离允许调用方用 `asyncio.wait_for` 包裹，或并行等待多个 actor 就绪（`asyncio.gather`）。

### Router 崩溃处理

`is_alive()` 方法供 `LLMServerManager` 定期探测。若 thread 退出，Manager 可以选择：
- **重启 Router**：`router_actor.start.remote()` 再调一次（Router 是无状态的，只要 replica 地址不变即可重启）
- **降级到直连**：触发告警并 fallback 到 `GlobalRequestLoadBalancer` 的 least-inflight 逻辑

当前 MVP 版本只做日志告警，不自动重启（避免引入复杂的重试状态机）。

---

## 测试要点

1. Router 启动后 `/health` 返回 200
2. `wait_until_ready(timeout=1)` 在 Router 未启动时抛出 `TimeoutError`
3. 向 Router 发送 `/generate` 请求能被正确转发到 mock worker
4. Actor 被 Ray kill 后 daemon thread 退出，不留僵尸端口
