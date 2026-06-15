# Dynamic Resource Scaling for Fully-Async Training

This module provides **hybrid inference resource dynamic scaling** for the fully-async training framework, enabling Trainer-node GPUs to participate in rollout generation during idle periods and thus improving overall GPU utilization.

---

## 1. Overview

### Problem

In the fully-async separated architecture, Trainer-node GPUs sit idle while waiting for rollout data, and Standalone Rollout nodes wait during training. This leads to suboptimal GPU utilization on both sides.

### Solution

This commit (`feat(dynamic-resource): init hybrid + standalone rollout resource with dynamic scaling`) introduces a **Hybrid + Standalone dual-mode inference resource** design:

- **Standalone replicas**: Always-on inference replicas on dedicated rollout nodes.
- **Hybrid replicas**: Inference replicas that share Trainer-node GPUs. They are activated during training idle time; before each training step, weights are offloaded and GPU memory is returned to the training engine.

`DynamicResourceController` manages the lifecycle of hybrid replicas. A pluggable **Policy** decides when to activate and deactivate:

```
State machine:  STANDALONE_ONLY  <->  HYBRID_ACTIVE

Activate (after weight sync):
  1. add_replicas               — register hybrid replicas in the load balancer
  2. resume_generation_replicas — allow hybrid replicas to accept requests

Deactivate (order is critical):
  1. remove_replicas  — cut routing first; prevents retry loop re-routing to dying replicas
  2. abort_replicas   — abort in-flight requests; partial-rollout retries go to standalone
  3. sleep_replicas   — release KV cache + offload weights, return GPU to training engine
```

### Policy Call Order Per Training Step

```
1. should_deactivate()          — before training; decide whether to deactivate hybrid replicas
2. deactivate_wait_samples()    — if (1) is True; return the minimum buffered-sample threshold
3. should_activate_after_step() — after weight sync; decide whether to (re-)activate hybrid replicas
4. request_rebalance()          — after activation; redistribute requests across replicas
5. update_after_step()          — after weight sync; update policy internal state
```

---

## 2. Configuration Parameters

Add the following fields under the `async_training` section of your training config (`fully_async_ppo_megatron_trainer.yaml`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dynamic_resource_scaling` | bool | `False` | Enable dynamic resource scaling. When `True`, hybrid rollout replicas are initialised on Trainer-node GPUs at startup (sleeping, memory returned to the training engine). |
| `dynamic_scaling_policy` | str | `"default"` | Name of the scaling policy to use. Built-in policies are described below. Custom policies can be registered with `@register_policy`. |
| `dynamic_scaling_deactivate_ratio` | float | `0.3` | Sample-collection ratio threshold used by the `default` policy (and typically custom policies too). The controller waits until `deactivate_ratio × required_samples × trigger_parameter_sync_step` samples are buffered before deactivating hybrid replicas. Lower values deactivate earlier; `1.0` waits for a full batch. |
| `staleness_threshold` | float | `0.1` | Allowed sample staleness ratio (existing config); affects `buffer_samples` computation. |

**Rollout resource config** (under `actor_rollout_ref.rollout`):

| Parameter | Description |
|-----------|-------------|
| `gpu_memory_utilization` | Memory utilization for hybrid replicas shared with the training engine. Keep low (e.g. 0.3–0.5) to leave room for training. |
| `standalone_gpu_memory_utilization` | Memory utilization for standalone replicas on dedicated rollout nodes (e.g. 0.6–0.8). Falls back to `gpu_memory_utilization` if `null`. |

**Standalone node config** (under `rollout`):

| Parameter | Description |
|-----------|-------------|
| `rollout.nnodes` | Number of standalone rollout nodes. Set to `0` to fall back to pure colocated mode (hybrid replicas only, no standalone nodes required). |

---

## 3. Built-in Demo Policies

### 3.1 `default` — DefaultDynamicScalingPolicy

**Registered name:** `"default"`  
**File:** `default_policy.py`

The recommended dynamic scaling policy with **adaptive deactivate_ratio**:

- **Deactivation logic**: Attempts to deactivate whenever hybrid replicas are active.
- **Wait threshold**: `deactivate_ratio × required_samples × trigger_parameter_sync_step`, ensuring enough samples are buffered before deactivation.
- **Adaptive ratio adjustment** (disabled when `only_hybrid=True`):
  - Trainer wait > 10 s → `ratio += 0.05` (rollout is the bottleneck; deactivate later)
  - Sample buffer excess → `ratio -= 0.05` (training is the bottleneck; deactivate earlier)

**Use case**: Hybrid + Standalone mixed deployment targeting maximum GPU utilization.

**Example scripts**: `examples/dynamic_scale/run_qwen35_35b_a3b_math_dynamic.sh`, `run_mimo_math_async_dynamic_static.sh`

---

### 3.2 `static_fully_async` — StaticFullyAsyncPolicy

**Registered name:** `"static_fully_async"`  
**File:** `static_fully_async_policy.py`

A special policy that is **fully equivalent to the original fully-async strategy**, designed for baseline comparisons or compatibility testing under the dynamic resource scaling framework.

| Method | Behaviour |
|--------|-----------|
| `should_deactivate()` | Returns `is_hybrid_active` (deactivate whenever active) |
| `deactivate_wait_samples()` | **Always returns `0`** (deactivate immediately, no waiting) |
| `should_activate_after_step()` | **Always returns `False`** (never re-activate after weight sync) |
| `update_after_step()` | No-op |

**Key properties:**

1. **Equivalent to standard Fully-Async**: Hybrid replicas are deactivated immediately (wait threshold = 0) at the start of each training step and are never re-activated. Trainer GPUs are always 100% returned to the training engine, producing the same behaviour as running without `use_dynamic_resource_scaling`.

2. **Degrades to Colocated mode**: When `rollout.nnodes=0` (standalone rollout resources set to 0), `only_hybrid=True` and the system automatically falls back to classic colocated mode — training and inference share the same GPUs, no separate rollout nodes are needed.

**Example scripts**: `examples/dynamic_scale/run_mimo_math_async_static_8_8.sh` (sets `use_dynamic_resource_scaling=False` to disable entirely), and scripts using `dynamic_scaling_policy="static_fully_async"`.

---

## 4. Adding a Custom Policy

Four steps to support a new policy:

### Step 1: Subclass the base class and implement required methods

```python
from verl.experimental.fully_async_policy.dynamic_scaling import (
    DynamicScalingPolicyBase,
    DynamicScaleContext,
    register_policy,
)

@register_policy("my_policy")
class MyDynamicScalingPolicy(DynamicScalingPolicyBase):

    def __init__(self, deactivate_ratio: float = 0.5, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid

    def should_deactivate(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScaleContext,
    ) -> bool:
        """Return True to deactivate hybrid replicas this step."""
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScaleContext) -> int:
        """Return the minimum buffered-sample count before deactivation proceeds."""
        return int(ctx.required_samples * ctx.trigger_parameter_sync_step * self.deactivate_ratio)

    def should_activate_after_step(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScaleContext,
    ) -> bool:
        """Return True to re-activate hybrid replicas after weight sync."""
        return ctx.total_generated_samples < ctx.expected_samples + ctx.buffer_samples

    # Optional: override to update internal state after each step
    def update_after_step(self, global_steps: int, ctx: DynamicScaleContext) -> None:
        pass

    # Optional: override to customise request redistribution after activation
    def request_rebalance(self, global_steps: int, ctx: DynamicScaleContext) -> None:
        pass
```

### Step 2: Ensure the file is imported at startup

Place the new policy file inside `dynamic_scaling/` and import it in `__init__.py`:

```python
# dynamic_scaling/__init__.py
from .my_policy import MyDynamicScalingPolicy
```

Alternatively, import the module manually in your training entry script to trigger `@register_policy` registration.

### Step 3: Reference the policy in your config

```yaml
async_training:
  use_dynamic_resource_scaling: True
  dynamic_scaling_policy: "my_policy"
  dynamic_scaling_deactivate_ratio: 0.5
```

### Step 4 (Optional): Use `DynamicScaleContext` fields for decision logic

`DynamicScaleContext` provides the following runtime information for policy decisions:

| Field | Description |
|-------|-------------|
| `required_samples` | Minimum samples per collection (`ppo_mini_batch_size × require_batches`) |
| `trigger_parameter_sync_step` | Number of collections before a weight-sync step |
| `total_generated_samples` | Cumulative rollout samples since training began |
| `expected_samples` | Theoretical samples needed up to the current sync step |
| `buffer_samples` | Allowed buffer headroom (`expected × staleness_threshold`) |
| `step_wait_times` | Per-collection wait times within the latest step (seconds) |
| `only_hybrid` | `True` when there are no standalone replicas |
| `last_activate_duration_s` | Duration of the last activate cycle (weight sync + onload), in seconds |
| `last_deactivate_duration_s` | Duration of the last deactivate cycle (offload), in seconds |

---

## 5. File Structure

```
dynamic_scaling/
├── __init__.py                    # Public exports
├── base.py                        # DynamicScalingPolicyBase ABC, DynamicScaleContext, registry
├── default_policy.py              # DefaultDynamicScalingPolicy (adaptive dynamic scaling)
├── static_fully_async_policy.py   # StaticFullyAsyncPolicy (equivalent to original fully-async / colocated fallback)
├── dynamic_resource_controller.py # DynamicResourceController (state machine + lifecycle management)
└── README.md                      # This document
```
