# Task 02：LLMServerManager 双轨改造

> 改动文件：`verl/workers/rollout/llm_server.py`
> 依赖：Task 01（SGLangRouterActor）、Task 05（config schema）

---

## 背景

`LLMServerManager` 当前只有一条数据路径：所有 `generate` 请求和所有控制平面操作（sleep/wake/update_weights 等）都直接操作 `server_handles`（Ray actor）。

引入 Router 后，需要拆分为两条路径：

- **数据平面**：`generate` 请求 → Router HTTP 地址（新增）
- **控制平面**：sleep/wake/update_weights/clear_kv_cache 等 → `server_handles` 直连（保留不变）

---

## 改动范围

### 1. `_init_global_load_balancer()` → 按配置条件启动 Router

**当前代码**（[llm_server.py:359-363](../../verl/workers/rollout/llm_server.py)）：

```python
async def _init_global_load_balancer(self) -> None:
    self.global_load_balancer = GlobalRequestLoadBalancer.remote(
        servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
    )
```

**改后**：

```python
async def _init_global_load_balancer(self) -> None:
    lb_config = getattr(self.rollout_config, "load_balance", None)
    use_router = (
        lb_config is not None
        and getattr(lb_config, "strategy", "least_inflight") == "sglang_router"
        and self.rollout_config.name == "sglang"
    )

    if use_router:
        await self._init_sglang_router(lb_config)
    else:
        # 原有逻辑，向后兼容
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        )
        self.router_address = None
        self.router_actor = None
```

### 2. 新增 `_init_sglang_router()`

```python
async def _init_sglang_router(self, lb_config) -> None:
    from verl.utils.net_utils import get_free_port
    from verl.workers.rollout.sglang_router_actor import SGLangRouterActor

    router_port = get_free_port()
    router_cfg = getattr(lb_config, "router", {})

    # 部署策略：STANDALONE 模式尽量调度到 rollout 节点
    actor_options = {"num_cpus": 1}
    if not self.worker_group:  # STANDALONE
        rollout_node_id = await self.server_handles[0].get_node_id.remote()
        actor_options["scheduling_strategy"] = (
            ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=rollout_node_id, soft=True
            )
        )

    self.router_actor = SGLangRouterActor.options(**actor_options).remote(
        worker_urls=self.server_addresses,
        host="0.0.0.0",
        port=router_port,
        policy=getattr(router_cfg, "policy", "cache_aware"),
        cache_threshold=getattr(router_cfg, "cache_threshold", 0.3),
        balance_abs_threshold=getattr(router_cfg, "balance_abs_threshold", 64),
        balance_rel_threshold=getattr(router_cfg, "balance_rel_threshold", 1.5),
        request_id_headers=["x-verl-request-id"],
        health_check_interval_secs=getattr(router_cfg, "health_check_interval_secs", 60),
    )
    await self.router_actor.start.remote()
    await self.router_actor.wait_until_ready.remote(timeout=120.0)
    self.router_address = await self.router_actor.get_address.remote()
    self.global_load_balancer = None  # 不再使用

    logger.info(f"LLMServerManager: Rust Router started at {self.router_address}")
```

### 3. `get_client()` 根据路由模式返回不同 client

**当前代码**（[llm_server.py:365-377](../../verl/workers/rollout/llm_server.py)）：

```python
def get_client(self, fully_async: bool = False) -> LLMServerClient:
    servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
    if not fully_async:
        return LLMServerClient(config=self.config, servers=servers, load_balancer_handle=self.global_load_balancer)
    else:
        return FullyLLMServerClient(...)
```

**改后**：

```python
def get_client(self, fully_async: bool = False) -> LLMServerClient:
    if self.router_address:
        # 方案 B：通过 Router HTTP 地址路由
        client_cls = FullyLLMServerClient if fully_async else LLMServerClient
        return client_cls(
            config=self.config,
            router_address=self.router_address,
            # 控制平面直连仍需 server_handles，但 client 不需要，由 Manager 自己持有
        )
    else:
        # 原有路径：GlobalRequestLoadBalancer
        servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
        client_cls = FullyLLMServerClient if fully_async else LLMServerClient
        return client_cls(
            config=self.config,
            servers=servers,
            load_balancer_handle=self.global_load_balancer,
        )
```

### 4. 控制平面方法保持不变

`clear_kv_cache()`、`start_profile()`、`stop_profile()` 全部操作 `self.rollout_replicas`（通过 `self.server_handles`），不经过 Router，**无需改动**。

---

## 新的 Manager 状态字段总览

| 字段 | 类型 | 路径 | 说明 |
|---|---|---|---|
| `rollout_replicas` | `list[RolloutReplica]` | 控制平面 | 保留不变 |
| `server_handles` | `list[ActorHandle]` | 控制平面 | 保留不变，各 replica node-0 server |
| `server_addresses` | `list[str]` | 控制平面 / Router 注册 | 保留不变 |
| `router_actor` | `SGLangRouterActor \| None` | 数据平面生命周期 | 新增，None 表示走旧路径 |
| `router_address` | `str \| None` | 数据平面 | 新增，Router 的 HTTP 地址 |
| `global_load_balancer` | `ActorHandle \| None` | 数据平面（旧） | 旧路径用，新路径为 None |

---

## 兼容性保证

- `load_balance.strategy` 不配置或为 `"least_inflight"` 时，走完全相同的旧路径
- `GlobalRequestLoadBalancer` 类不删除，旧路径零改动
- `LLMServerClient.__init__` 需要支持两种模式（见 Task 03）

---

## 边界情况处理

### Router 启动失败

`wait_until_ready()` 超时抛出 `TimeoutError`，由 `LLMServerManager.create()` 的 `asyncio.gather` 自然传播，训练启动失败并打印清晰错误信息。不静默降级（避免用户不知道 Router 没生效）。

### HYBRID 模式下 Router 的节点

HYBRID 模式下 rollout 和 trainer 共用节点，Router 运行在同一物理机，loopback 访问延迟 < 0.5ms，可接受。

### 多个 `LLMServerClient` 共享同一 Router

`router_address` 是一个字符串，`get_client()` 每次返回新的 `LLMServerClient` 对象，但它们都指向同一个 Router HTTP 端点。Router 本身是并发安全的（Rust async），不存在共享状态竞争。
