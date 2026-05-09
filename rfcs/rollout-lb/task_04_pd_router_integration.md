# Task 04：PD 模式 Router 集成

> 改动文件：`verl/workers/rollout/sglang_rollout/sglang_pd_replica.py`，`verl/workers/rollout/llm_server.py`
> 依赖：Task 01、Task 02

---

## 背景

### 当前 PD decode 选择逻辑

[sglang_pd_replica.py](../../verl/workers/rollout/sglang_rollout/sglang_pd_replica.py) 中，`SGLangPDReplica` 在 `launch_servers()` 完成后，decode peer 的选择发生在 `SGLangHttpServer.generate()` 内部（[async_sglang_server.py:496-520](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py)）：

```python
# async_sglang_server.py:496-520（prefill server 的 generate）
if self._disaggregation_role == "prefill" and self._pd_decode_peers and bootstrap_room is None:
    room = secrets.randbits(63)
    decode_peer = self._pd_decode_peers[secrets.randbelow(len(self._pd_decode_peers))]
    # 并发发送 prefill 和 decode 请求
    _, decode_output = await asyncio.gather(prefill_coro, decode_coro)
    return decode_output
```

这意味着 decode peer 选择在 **sglang 进程内部**随机，verl 的 `GlobalRequestLoadBalancer` 对此完全不感知。

### Rust Router PD 模式能提供什么

Rust Router 的 PD 模式（`pd_disaggregation=True`）支持在 **Router 层面**协调 prefill+decode 配对，并为各角色独立配置路由策略：

```
Client
  │
  ▼
Rust Router (pd_disaggregation=True)
  │ prefill_policy=cache_aware
  ├──► prefill-0 (bootstrap 9001)
  │ decode_policy=power_of_two
  ├──► decode-0
  └──► decode-1
```

Router 会：
1. 根据 prefill_policy 选一个 prefill server
2. 根据 decode_policy 选一个 decode server
3. 协调两者通过 bootstrap 握手（KV cache 传输）
4. 把 decode server 的生成结果返回给 Client

---

## 当前 PD 架构与 Rust Router PD 的差异

| 维度 | 当前实现 | Rust Router PD |
|---|---|---|
| 路由发起者 | prefill server 内部（secrets.randbelow） | Router（应用层外） |
| bootstrap 协调 | prefill server 直接持有 decode actor handle，通过 `asyncio.gather` 并发 | Router 负责协调，通过 bootstrap URL 参数 |
| bootstrap 端口 | 预先 `get_free_port()` 分配，存在 TOCTOU 窗口（已有 sock 保活逻辑） | Router 在 prefill_urls 中接收 bootstrap_port |
| decode 选择策略 | 纯随机 | power_of_two / round_robin 等 |

---

## 方案设计

### 核心思路

PD 模式下，Router 的工作原理不同于 non-PD 模式：

- **Non-PD**：Router 透明转发 `/generate` 到某个 worker
- **PD**：Router 将一个 `/generate` 请求拆分为两个：一个发到 prefill server（携带 bootstrap 参数），一个发到 decode server（携带相同 bootstrap_room），由它们通过 NIXL/mooncake 完成 KV 传输

verl 当前已经在 `SGLangHttpServer.generate()` 内部实现了这个拆分逻辑（`asyncio.gather(prefill_coro, decode_coro)`）。**如果引入 Rust Router PD 模式，这套内部逻辑就要被 Router 接管**，两者不能并存。

因此有两种子方案：

---

### 子方案 A：Router 接管 PD 协调（完整集成）

**prefill server 暴露标准 `/generate` 端点**，不再内部选 decode peer。Router 负责选 prefill+decode 配对，并注入 bootstrap 参数。

**改动**：
1. `SGLangHttpServer.generate()` 的 PD 分支（`if self._disaggregation_role == "prefill" and ...`）移除，改为 prefill server 仅处理自己的 prefill 请求（带 bootstrap 参数则 KV 传输，否则完整生成）
2. `LLMServerManager._init_sglang_router()` 在 PD 模式下使用 PD 参数

**Router 注册**（在 `LLMServerManager._init_sglang_router()` 中）：

```python
if disagg_enabled:
    # 从各 replica 收集 prefill/decode 地址
    prefill_urls = []
    decode_urls = []
    for replica in self.rollout_replicas:
        pd_replica: SGLangPDReplica = replica
        prefill_urls.append((pd_replica._prefill_server_address, pd_replica._bootstrap_port))
        decode_urls.extend(pd_replica._decode_server_addresses)

    router_actor = SGLangRouterActor.options(...).remote(
        worker_urls=[],
        host="0.0.0.0",
        port=router_port,
        pd_disaggregation=True,
        prefill_urls=prefill_urls,    # [(addr, bootstrap_port), ...]
        decode_urls=decode_urls,      # [addr, ...]
        prefill_policy=getattr(router_cfg, "prefill_policy", "cache_aware"),
        decode_policy=getattr(router_cfg, "decode_policy", "power_of_two"),
    )
```

**问题**：Rust Router 的 PD 协调需要 prefill server 暴露 bootstrap 信息，而当前 verl 的 bootstrap 是在 `sglang` 进程内部管理的（`ServerArgs.disagg_bootstrap_port`）。需要确认 sglang `/generate` HTTP 端点是否支持外部传入 `bootstrap_room`（而不是 prefill server 内部 mint）。

从 [async_sglang_server.py:577-582](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py) 可见，sglang 的 HTTP `/generate` 端点**已支持**接受外部传入的 bootstrap 参数：

```python
if bootstrap_room is not None:
    request["bootstrap_host"] = bootstrap_host
    request["bootstrap_port"] = bootstrap_port
    request["bootstrap_room"] = bootstrap_room
```

这说明子方案 A 技术上可行。

**风险**：Rust Router 的 PD 协调逻辑是否与 verl 的 bootstrap 握手协议完全兼容，需要实测验证。

---

### 子方案 B：Router 只做 prefill 路由，decode 保持原有内部选择（最小改动）

**保留** `SGLangHttpServer.generate()` 内部的 decode peer 选择逻辑不变，Router 只负责从多个 prefill server 中选一个。Client 只发一个请求给 Router，Router 转发给某个 prefill server，prefill server 再内部选 decode peer。

这相当于 non-PD 模式的 Router，只是 worker_urls 只注册 prefill server 地址：

```python
# PD 模式下，Router 只知道 prefill servers
prefill_addresses = [replica._prefill_server_address for replica in pd_replicas]
router_actor = SGLangRouterActor(...).remote(
    worker_urls=prefill_addresses,   # 只注册 prefill
    policy="cache_aware",            # 对 prefill 做 cache-aware 路由
    pd_disaggregation=False,         # 不用 Router 的 PD 模式
)
```

**优点**：改动最小，不需要修改 `SGLangHttpServer.generate()` 的 PD 逻辑
**缺点**：decode 选择仍然是随机，无法获得 `power_of_two` 的均衡效果

---

### 推荐

MVP 阶段采用**子方案 B**，原因：
1. 改动风险低，不需要验证 Rust Router PD 协调与 verl bootstrap 的兼容性
2. prefill 侧获得 `cache_aware` 路由，这是最大的收益点（RLHF 系统 prompt 命中）
3. decode 均衡可以在子方案 A 作为后续优化

完整的子方案 A 作为 Phase 3 目标。

---

## 子方案 B 实现

### `_prefill_server_address` 字段暴露

`SGLangPDReplica` 当前已有 `_prefill_server_address` 字段（[sglang_pd_replica.py:145](../../verl/workers/rollout/sglang_rollout/sglang_pd_replica.py)），但 `RolloutReplica` 基类的 `_server_address` 也指向 prefill server。

需要在 `LLMServerManager` 中能区分 PD replica 和普通 replica，以收集正确的 prefill 地址用于 Router 注册。

```python
# LLMServerManager._init_sglang_router() 中
from verl.workers.rollout.sglang_rollout.sglang_pd_replica import SGLangPDReplica

if isinstance(self.rollout_replicas[0], SGLangPDReplica):
    # PD 模式：Router 只注册 prefill server 地址
    worker_urls_for_router = [
        r._prefill_server_address for r in self.rollout_replicas
    ]
else:
    # 普通模式：所有 replica 地址
    worker_urls_for_router = self.server_addresses
```

`self.server_addresses` 在 PD 模式下就是 `prefill_server_address`（replica 的 `_server_address` 指向 prefill），所以这里**实际上不需要特殊处理**——`self.server_addresses` 在两种模式下都是正确的注册地址。

---

## 对 `SGLangHttpServer.generate()` 的影响

子方案 B 下：**零改动**。prefill server 收到来自 Router 的 `/generate` 请求后，内部逻辑不变，仍然用 `secrets.randbelow` 选 decode peer。

唯一需要确认的是：Router 转发请求时是否保留了 `rid`（request id）字段，避免 sglang 因 rid 重复而报错。

Router 转发时会生成新的 HTTP 请求，`rid` 字段在 JSON body 中透明传递。verl 在 `_generate_via_router()` 构造 payload 时已经使用 `uuid4().hex` 作为 `rid`，不会重复。

---

## 未来 Phase 3（子方案 A）需要额外解决的问题

1. **Rust Router PD bootstrap 协议**：确认 Router 的 `/generate` → prefill 请求中注入的 `bootstrap_room` 格式与 sglang 期望的格式一致
2. **多 prefill replica 场景**：当前 `prefill_replicas > 1` 抛 `NotImplementedError`，Phase 3 可能需要同步解锁这个限制
3. **bootstrap_port 的生命周期**：子方案 A 下 Router 需要在路由时知道每个 prefill server 的 bootstrap_port，这个端口当前在 `SGLangPDReplica.launch_servers()` 动态分配，需要在 Router 注册时传递
