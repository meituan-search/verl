# Task 05：Config Schema 设计

> 改动文件：`verl/trainer/config/rollout/rollout.yaml`，`verl/workers/config.py`
> 依赖：无（最先确认，其他 task 参照此文档）

---

## 设计原则

1. **向后兼容**：不配置 `load_balance` 节时，行为与现有完全相同（least_inflight + LRU sticky）
2. **sglang 专属**：Router 策略只对 `rollout.name=sglang` 生效；其他引擎（vllm/trtllm/hf）忽略此配置
3. **渐进开放**：先只支持 `least_inflight` 和 `sglang_router` 两种策略，后续再扩展

---

## rollout.yaml 新增节

在 `verl/trainer/config/rollout/rollout.yaml` 的 `disaggregation` 节之后追加：

```yaml
# Load balance configuration for multi-replica rollout servers.
# Only effective when rollout.name=sglang and multiple replicas are used.
load_balance:

  # Required when using verl.utils.omega_conf_to_dataclass to instantiate dataclass configs
  _target_: verl.workers.config.LoadBalanceConfig

  # Load balancing strategy.
  # - least_inflight: default, least in-flight requests (current behavior)
  # - sglang_router: delegate to sgl-model-gateway Rust Router (requires sglang_router package)
  strategy: least_inflight

  # sglang_router strategy settings (only used when strategy=sglang_router)
  router:

    # Required when using verl.utils.omega_conf_to_dataclass to instantiate dataclass configs
    _target_: verl.workers.config.SGLangRouterConfig

    # Routing policy passed to sgl-model-gateway.
    # Options: cache_aware, round_robin, consistent_hashing, prefix_hash
    # For PD mode, this is the prefill policy; decode policy is set separately below.
    policy: cache_aware

    # cache_aware: route to replica if prefix match rate > this threshold.
    cache_threshold: 0.3

    # cache_aware: trigger load balancing if (max_load - min_load) > this value.
    balance_abs_threshold: 64

    # cache_aware: trigger load balancing if max_load > min_load * this value.
    balance_rel_threshold: 1.5

    # How often the Router checks replica health (seconds).
    health_check_interval_secs: 60

    # PD disaggregation decode policy (only used when disaggregation.enabled=True).
    # Options: power_of_two, round_robin, random
    # power_of_two: pick best of 2 random decode workers by in-flight count.
    decode_policy: power_of_two
```

---

## Python dataclass 定义

在 `verl/workers/config.py` 中新增两个 dataclass（与现有 `RolloutConfig`、`DisaggregationConfig` 风格一致）：

```python
@dataclasses.dataclass
class SGLangRouterConfig:
    """Configuration for sgl-model-gateway Rust Router."""
    policy: str = "cache_aware"
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 64
    balance_rel_threshold: float = 1.5
    health_check_interval_secs: int = 60
    decode_policy: str = "power_of_two"   # PD mode decode strategy


@dataclasses.dataclass
class LoadBalanceConfig:
    """Load balancing configuration for multi-replica rollout."""
    strategy: str = "least_inflight"
    router: SGLangRouterConfig = dataclasses.field(default_factory=SGLangRouterConfig)
```

---

## `RolloutConfig` 新增字段

```python
@dataclasses.dataclass
class RolloutConfig:
    # ... 现有字段 ...
    load_balance: LoadBalanceConfig = dataclasses.field(default_factory=LoadBalanceConfig)
```

---

## 读取方式（在 LLMServerManager 中）

```python
lb_config: LoadBalanceConfig = self.rollout_config.load_balance
use_router = (
    lb_config.strategy == "sglang_router"
    and self.rollout_config.name == "sglang"
)
```

---

## 典型用户配置示例

### 启用 cache_aware 路由（RLHF，system prompt 重复）

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 4
    data_parallel_size: 4          # 4 replicas
    load_balance:
      strategy: sglang_router
      router:
        policy: cache_aware
        cache_threshold: 0.3
```

### PD 模式 + decode power_of_two

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 4
    disaggregation:
      enabled: True
      prefill_replicas: 1
      decode_replicas: 3
    load_balance:
      strategy: sglang_router
      router:
        policy: cache_aware        # prefill 路由策略
        decode_policy: power_of_two
```

### 保持默认行为（不填 load_balance）

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 4
    # load_balance 不配置 → 自动使用 least_inflight，行为与旧版完全一致
```

---

## 验证逻辑

在 `LLMServerManager.__init__` 或 `_init_global_load_balancer()` 中加入配置校验：

```python
lb_config = self.rollout_config.load_balance

if lb_config.strategy == "sglang_router":
    # 确认 sglang_router 包可用
    try:
        import sglang_router  # noqa: F401
    except ImportError:
        raise ImportError(
            "load_balance.strategy=sglang_router requires the sglang_router package. "
            "Install it with: pip install sglang[router]"
        )
    # 确认 rollout 引擎是 sglang
    if self.rollout_config.name != "sglang":
        raise ValueError(
            f"load_balance.strategy=sglang_router is only supported for sglang rollout, "
            f"but rollout.name={self.rollout_config.name!r}"
        )

VALID_STRATEGIES = {"least_inflight", "sglang_router"}
if lb_config.strategy not in VALID_STRATEGIES:
    raise ValueError(
        f"load_balance.strategy={lb_config.strategy!r} is not supported. "
        f"Valid options: {VALID_STRATEGIES}"
    )
```

---

## 与现有配置的关系

| 现有字段 | 与 load_balance 的关系 |
|---|---|
| `rollout.data_parallel_size` | 决定 replica 数量，>1 时 load balance 才有意义 |
| `disaggregation.enabled` | PD 模式下 `router.decode_policy` 生效 |
| `rollout.enable_prefix_caching` | 应设为 True 才能让 cache_aware 路由有效，两者独立但互补 |
| `prometheus.enable` | Router 自身也支持 Prometheus（`router.prometheus_port`），暂不在此 RFC 范围内 |
