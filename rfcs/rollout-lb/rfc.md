# 方案 B：引入 sglang Rust Router 作为 verl 内嵌 sidecar

> 状态：设计分析阶段
> 关联 RFC：[rollout 侧 load balance 总览](./RFC_总览.md)

---

## 一、背景：现状与问题

### 现有负载均衡机制

verl 的 rollout 侧多副本管理通过 `GlobalRequestLoadBalancer` Ray actor 实现（[verl/workers/rollout/llm_server.py](../../verl/workers/rollout/llm_server.py)）：

- **新请求**：least-inflight——路由到当前 in-flight 请求数最少的 replica
- **多轮对话**：sticky session——LRU cache（`request_id → server_id`），同一对话复用同一 replica

### 问题

1. **In-flight 计数不反映实际负载**：sglang 内部 DP 并行和 KV cache 利用率差异被忽略
2. **无 prefix cache 感知**：RLHF 场景下系统 prompt 大量重复，cache-aware 路由能显著降低 TTFT
3. **PD decode 选择纯随机**：`SGLangPDReplica` 中 decode peer 用 `secrets.randbelow`，未考虑各 decode server 队列深度
4. **无健康检查**：故障节点不会被自动排除出路由候选

### 已有铺垫（无需从零开发）

- **[http_server_engine.py:204-286](../../verl/workers/rollout/sglang_rollout/http_server_engine.py)**：`HttpServerAdapter` 已实现 `_register_with_router()` 和 `shutdown()` 中的注销逻辑（`/add_worker`/`/remove_worker`），仅被 TODO 注释挡住：
  ```python
  # TODO: @ChangyiYang Enable SGLang router for this http server engine
  ```
- **[router_args.py](../../third_party/sglang/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py)**：所有 Router 参数已有 `RouterArgs` dataclass，可直接 `Router.from_args()` 启动

---

## 二、方案核心思路

引入 `sgl-model-gateway` Rust Router 作为 verl 内嵌 sidecar（Ray actor 封装）：

- **数据平面**（generate）：所有请求经 Router 路由，Router 负责 cache_aware / round_robin / power_of_two 等策略
- **控制平面**（sleep/wake/update_weights 等）：绕过 Router，仍直连各 replica Ray actor

---

## 三、架构变化

### 当前架构

```
AgentLoopWorker
    │  generate(request_id, ...)
    ▼
LLMServerClient
    │  acquire_server(request_id) ──► GlobalRequestLoadBalancer [Ray actor]
    │                                      │ least-inflight + LRU sticky
    │  ◄── server_id + handle             ▼
    │                            in-flight counter dict
    │
    └──► SGLangHttpServer.generate.remote(...)  [Ray actor]
              │
              ▼
         HTTP POST /generate  →  sglang engine
```

### 引入 Rust Router 后

```
AgentLoopWorker
    │  generate(request_id, ...)
    ▼
LLMServerClient
    │  Header: x-verl-request-id: {request_id}
    └──► aiohttp POST http://router_host:router_port/generate
                  │
        [sgl-model-gateway Rust Router]  ← SGLangRouterActor [Ray]
          cache_aware / round_robin / power_of_two
          健康检查 + circuit breaker
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    replica-0  replica-1  replica-2
    (SGLangHttpServer HTTP 端口)

控制平面（保持直连）：
LLMServerManager → server_handles[i].sleep/wake/update_weights/...
```

### LLMServerManager 双轨数据结构

```python
class LLMServerManager:
    router_address: str              # generate 路径（新增）
    router_actor: SGLangRouterActor  # Router 生命周期管理（新增）
    server_handles: list             # 控制平面直连（保留）
    server_addresses: list           # 控制平面直连（保留）
```

---

## 四、关键集成点逐一分析

### 4.1 Router 进程启动与生命周期

**启动位置**：`LLMServerManager._init_global_load_balancer()` 改为条件性启动 Router。

**Ray 封装**（Rust Router 是 blocking server，需 daemon thread）：

```python
@ray.remote(num_cpus=1)
class SGLangRouterActor:
    def __init__(self, args: RouterArgs):
        from sglang_router.router import Router
        self._router = Router.from_args(args)
        self._addr = f"http://{args.host}:{args.port}"

    def start(self):
        import threading
        t = threading.Thread(target=self._router.start, daemon=True)
        t.start()

    def get_address(self) -> str:
        return self._addr
```

**worker 注册方式**：使用 `RouterArgs(worker_urls=[addr0, addr1, ...])` 一次性预注册（所有 replica 在 `_initialize_llm_servers()` 完成后地址已知），无需动态 `/add_worker`。

**端口分配**：调用 `get_free_port()`（已在 `verl/utils/net_utils.py`）。

**部署位置**：STANDALONE 模式下，`SGLangRouterActor` 应通过 Ray 调度约束到 rollout 节点，减少跨节点 HTTP 跳数。

---

### 4.2 Sticky Session 语义变化

| | 当前 | 方案 B |
|---|---|---|
| 机制 | 应用层 LRU cache（`request_id → replica`） | `cache_aware` prefix match 隐式 affinity |
| 多轮对话 | 强 sticky，LRU 保证同一 replica | 每次带完整前缀，prefix match 自然路由回有 KV cache 的 replica |
| 实现层 | `GlobalRequestLoadBalancer` Python actor | Rust Router 内部 radix tree |

**关键结论**：`FullyLLMServerClient` partial rollout 场景中，每次调用都带完整已生成前缀（`prompt_ids + final_output.token_ids`，[llm_server.py:207](../../verl/workers/rollout/llm_server.py) 已如此实现），`cache_aware` 路由会自然将续写请求路由回有 KV cache 的 replica。**这比应用层 LRU 更优雅**——不需要维护 session 状态，prefix cache 本身就是最好的 affinity 信号。

**注意**：`clear_kv_cache` 调用后，下一轮请求的 sticky 失效（KV cache 本身被清空，这是预期行为，两种方案行为一致）。

---

### 4.3 Request ID 透传

Router 配置：`request_id_headers: ["x-verl-request-id"]`

`LLMServerClient.generate()` 需在 HTTP header 中注入：
```python
headers = {"x-verl-request-id": request_id}
await session.post(router_url, json=payload, headers=headers)
```

---

### 4.4 控制平面操作（不经 Router）

以下操作**必须绕过 Router**，直连各 replica：

| 操作 | 调用位置 |
|---|---|
| `sleep()` / `wake_up()` | `SGLangReplica`，training 期间 GPU 内存释放/恢复 |
| `update_weights()` | `sglang_rollout.py:update_weights()`，模型权重同步（CUDA IPC） |
| `clear_kv_cache()` | `LLMServerManager.clear_kv_cache()`，每轮 rollout 前 |
| `abort_all_requests()` | partial rollout 机制 |
| `resume_generation()` | partial rollout 机制 |
| `set_global_steps()` | async training，模型版本追踪 |
| `load_lora_adapter_from_tensor()` | LoRA 权重热更新 |

`LLMServerManager` 必须同时维护 `server_handles`（Ray actor），保证这些操作路径不受 Router 影响。

---

### 4.5 PD 模式集成

Rust Router 原生支持 PD disaggregated 模式（`pd_disaggregation=True`），可分别配置 prefill/decode 策略。

**注册格式**：
```python
RouterArgs(
    pd_disaggregation=True,
    prefill_urls=[(prefill_addr, bootstrap_port)],
    decode_urls=[decode_addr_0, decode_addr_1, ...],
    prefill_policy="cache_aware",
    decode_policy="power_of_two",  # 替代当前 secrets.randbelow
)
```

`power_of_two`：随机选 2 个 decode worker，取 in-flight 较少的那个（Power of Two Choices）。

---

### 4.6 接口层变化（最大风险点）

`LLMServerClient.generate()` 从 **Ray remote call** 改为 **aiohttp HTTP call**：

- 影响 `TokenOutput` 的序列化/反序列化（当前 Ray 序列化自动处理 Pydantic model，改为 HTTP 需手动 JSON 解析）
- 影响错误处理语义（Ray exception vs HTTP status code）
- 影响 `rollout_trace_op` 装饰器的 span 上下文传播
- `LLMServerClient` 不再持有 `SGLangHttpServer` Ray actor handle

---

## 五、Rust Router 策略参考

| 策略 | 字符串 | 适用场景 |
|---|---|---|
| `cache_aware`（**推荐默认**） | `"cache_aware"` | RLHF，system prompt 大量重复，prefix cache 命中率高 |
| `round_robin` | `"round_robin"` | 均匀分发，无状态场景 |
| `power_of_two` | `"power_of_two"` | PD decode 均衡（PD 专用） |
| `consistent_hashing` | `"consistent_hashing"` | 更强 cache locality |
| `prefix_hash` | `"prefix_hash"` | 按 prefix hash 路由 |

**cache_aware 关键参数**：
- `cache_threshold=0.3`：prefix 匹配率阈值，超过则路由到命中 replica
- `balance_abs_threshold=64`：负载差超过此值触发负载均衡覆盖 cache 路由
- `balance_rel_threshold=1.5`：负载比超过此值触发负载均衡

**健康检查**：每 60s 请求 `/health`，连续 3 次失败→标记不健康，排除路由；连续 2 次成功→恢复。
**Circuit breaker**：120s 窗口内 10 次失败→开路，60s 后半开探测。

---

## 六、优缺点总结

### 优点

1. **cache_aware 路由**：prefix cache 命中率提升，TTFT 改善（RLHF system prompt 重复场景收益最大）
2. **健康检查 + circuit breaker**：开箱即用，当前完全缺失
3. **PD decode power_of_two**：替代纯随机，统计意义上更均衡
4. **消除 GlobalRequestLoadBalancer 单点瓶颈**：Rust Router 异步并发，无 Ray actor 的 acquire/release RPC 开销
5. **sticky 语义更优雅**：依赖 prefix cache 而非应用层状态维护

### 缺点

1. **额外 HTTP hop 延迟**：同节点 ~0.1-0.5ms，跨节点 ~1-5ms（LLM 生成秒级耗时，通常可接受）
2. **双轨架构**：数据平面走 Router，控制平面直连，增加维护认知负担
3. **Router 进程脆弱性**：Rust 进程崩溃需重启逻辑，比 Python Ray actor 复杂
4. **接口层变化影响面大**：`LLMServerClient.generate()` Ray→HTTP 改动，涉及序列化、错误处理、tracing
5. **额外 TCP 端口**：K8s/Slurm 严格网络环境需额外配置
6. **sglang_router 包安装**：需要 Rust binding，与 verl 现有 `uv pip install` 流程需额外集成

---

## 七、改动范围评估

| 文件 | 改动内容 | 风险 |
|---|---|---|
| `verl/workers/rollout/llm_server.py` | `LLMServerClient.generate()` 改 HTTP call；`LLMServerManager` 双轨结构；`_init_global_load_balancer()` 条件启动 Router | **高** |
| `verl/workers/rollout/sglang_rollout/http_server_engine.py` | 解除 TODO，激活 `_register_with_router()` | 低 |
| `verl/workers/rollout/sglang_rollout/sglang_pd_replica.py` | PD 模式 Router 注册，`_select_decode_peer()` 逻辑移除 | 中 |
| 新文件：`verl/workers/rollout/sglang_router_actor.py` | `SGLangRouterActor` Ray 封装 | 低 |
| `verl/trainer/config/rollout/rollout.yaml` | 新增 `load_balance.strategy`、`load_balance.router.*` 配置项 | 低 |
| 测试 | Router 进程管理、故障恢复、策略切换 | 大 |

---

## 八、建议实施路径

- **Phase 1（低风险过渡）**：在 `GlobalRequestLoadBalancer` 内增加 `/v1/loads` 轮询（`least_load` 策略）+ health check，不涉及接口层变化
- **Phase 2（方案 B 主体）**：`SGLangRouterActor` + `cache_aware` 策略，`LLMServerClient` 改 HTTP call
- **Phase 3（PD 模式）**：Router PD 模式集成，`power_of_two` decode 均衡
