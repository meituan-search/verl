# Load Balance 使用说明

> 适用版本：实现方案 B 后

---

## 一、快速上手

### 默认行为（无需任何配置）

不配置 `load_balance` 节时，verl 使用原有的 **least-inflight + LRU sticky session** 策略，行为与引入此功能前完全一致。

```yaml
# 无需任何 load_balance 配置，默认生效
actor_rollout_ref:
  rollout:
    name: sglang
    data_parallel_size: 4
```

---

### 启用 Rust Router（cache_aware 策略）

适用场景：**多副本 RLHF 训练**，系统 prompt 大量重复，希望最大化 prefix cache 命中率、降低 TTFT。

**前置条件**：安装 `sglang_router` Python 包（Rust binding）：

```bash
pip install sglang[router]
# 或从源码安装（third_party/sglang/sgl-model-gateway）：
pip install third_party/sglang/sgl-model-gateway/bindings/python
```

**配置**：

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 4
    data_parallel_size: 4        # 4 个 replica，LB 才有意义
    enable_prefix_caching: True  # 与 cache_aware 路由配合使用

    load_balance:
      strategy: sglang_router
      router:
        policy: cache_aware
```

---

## 二、所有配置项

```yaml
load_balance:
  # 路由策略，必填（默认 least_inflight）
  # - least_inflight : 最少 in-flight 请求 + LRU sticky session（原有行为）
  # - sglang_router  : 委托给 sgl-model-gateway Rust Router
  strategy: least_inflight

  # 以下 router.* 仅在 strategy=sglang_router 时生效
  router:
    # 路由策略，传给 Rust Router
    # - cache_aware       : prefix cache 感知路由（推荐，RLHF 场景默认）
    # - round_robin       : 轮询
    # - consistent_hashing: 一致性哈希
    # - prefix_hash       : 按 prefix hash 路由
    policy: cache_aware

    # cache_aware 参数
    # prefix 匹配率超过此阈值时，路由到命中 replica
    cache_threshold: 0.3

    # 负载差 (max_load - min_load) 超过此值时，改为负载均衡路由
    balance_abs_threshold: 64

    # max_load > min_load * 此值时，改为负载均衡路由
    balance_rel_threshold: 1.5

    # Rust Router 对各 replica 进行健康检查的间隔（秒）
    health_check_interval_secs: 60

    # PD 模式 decode pool 的路由策略（仅 disaggregation.enabled=True 时有意义）
    # - power_of_two: 随机选 2 个 decode worker，取负载较低的（推荐）
    # - round_robin : 轮询
    # - random      : 纯随机（等价于当前行为）
    decode_policy: power_of_two
```

---

## 三、典型配置示例

### 场景 1：大规模 RLHF，system prompt 重复率高

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 8
    data_parallel_size: 8
    enable_prefix_caching: True

    load_balance:
      strategy: sglang_router
      router:
        policy: cache_aware
        cache_threshold: 0.3      # 命中率 > 30% 即路由到命中 replica
        balance_abs_threshold: 64
        balance_rel_threshold: 1.5
```

### 场景 2：多副本均匀分发，无状态场景（如 reward model）

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    data_parallel_size: 4

    load_balance:
      strategy: sglang_router
      router:
        policy: round_robin
```

### 场景 3：PD 模式，多个 decode replica 均衡

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 4
    enable_prefix_caching: True

    disaggregation:
      enabled: True
      prefill_replicas: 1
      decode_replicas: 4

    load_balance:
      strategy: sglang_router
      router:
        policy: cache_aware          # prefill 侧：prefix cache 感知
        decode_policy: power_of_two  # decode 侧：pick-of-2 均衡
```

### 场景 4：保持原有行为（默认，无需显式配置）

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    data_parallel_size: 4
    # 不配置 load_balance，或显式写：
    load_balance:
      strategy: least_inflight
```

---

## 四、注意事项

### 4.1 前置依赖

`strategy: sglang_router` 要求：
1. `rollout.name` 必须为 `sglang`（vllm/trtllm 不支持）
2. 安装 `sglang[router]` Python 包（含 Rust binding `sglang_router_rs`）
3. Router 需要占用一个额外的 TCP 端口（自动分配，无需手动指定）

配置错误时，启动阶段会报错并说明原因，不会静默降级：

```
ImportError: load_balance.strategy=sglang_router requires the sglang_router package.
Install it with: pip install sglang[router]
```

### 4.2 data_parallel_size=1 时无效

单 replica 情况下所有策略等效，Router 只有一个后端，路由没有意义。建议在 `data_parallel_size > 1` 或多节点 standalone 部署时才启用。

### 4.3 enable_prefix_caching 配合使用

`cache_aware` 路由依赖 sglang 的 prefix cache 命中信息，与 `rollout.enable_prefix_caching: True` 配合使用效果最佳。两者独立控制，但需同时开启才能发挥 `cache_aware` 的优势。

### 4.4 Sticky session 语义变化

启用 `sglang_router` 后，多轮对话的 sticky session 由 **Rust Router 的 cache_aware prefix match** 实现（而非原有的应用层 LRU cache）。行为差异：

| | least_inflight（原有） | sglang_router |
|---|---|---|
| 多轮对话路由 | LRU cache，强 sticky | prefix match，隐式 affinity |
| 每轮清空 KV cache 后 | sticky 失效（预期行为） | 同上（一致） |
| partial rollout resume | 同 replica | prefix 匹配自动路由回原 replica |

对绝大多数场景，两者行为等效，`sglang_router` 模式在 prefix cache 命中率上更优。

### 4.5 控制平面操作不受影响

`sleep()`、`wake_up()`、`update_weights()`、`clear_kv_cache()` 等 training-rollout 交互操作全部走直连路径（不经 Router），与 `load_balance` 配置无关。

---

## 五、故障排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| 启动时 `ImportError: sglang_router` | 未安装 `sglang[router]` | `pip install sglang[router]` |
| 启动时 `TimeoutError: Router not ready` | Router 端口被占用或网络隔离 | 检查防火墙/K8s NetworkPolicy；查看 Router 进程日志 |
| 请求 P99 延迟升高 | Router 跨节点部署引入额外 RTT | 确认 Router actor 调度在 rollout 节点（STANDALONE 模式自动处理） |
| 路由不均匀（某 replica 过热） | `cache_threshold` 过高导致过多路由到同一 replica | 调低 `cache_threshold`（如 0.1）或切换到 `round_robin` |
| `strategy=sglang_router` 配置未生效 | `rollout.name` 不是 `sglang` | 确认 `actor_rollout_ref.rollout.name: sglang` |
