# model_engine_server 对比实验

## 实验目标

验证 `model_engine_server` 在 fully_async 训练框架下的价值：对比不同 `old_log_prob` 计算方案在**训练效率**和**训练效果**上的差异。

---

## 背景：old_log_prob 的三种处理方式

PPO 重要性采样比率 = `exp(new_log_prob - old_log_prob)`。fully_async 解耦 rollout 与 training 后，训练时模型参数已更新，需要"生成时参数下的 log prob"。

| 方案 | 机制 | 关键配置 |
|------|------|----------|
| **不重算（bypass）** | 直接复用 vLLM 生成时顺带算的 `rollout_log_probs` | `bypass_mode=True` |
| **CPU 重算** | trainer 将旧参数存至 CPU RAM，切换回旧参数做 forward，再恢复 | `bypass_mode=False`, `enable_standalone=False` |
| **Server 重算** | 独立专用 GPU 节点异步并行计算，trainer 零等待 | `bypass_mode=False`, `enable_standalone=True` |

---

## 实验设置

**模型**：Qwen2.5-Math-7B  
**算法**：DAPO（GRPO + overlong buffer）  
**数据**：dapo-math-17k 训练，AIME 2024 评测  
**后端**：Megatron（TP=4, PP=2）+ vLLM  
**序列长度**：prompt 2k + response 28k = 30k tokens

**公共参数**（4 组实验完全一致）：

```
max_prompt_length=2048, max_response_length=28672
n_resp_per_prompt=16, ppo_mini_batch_size=32
use_dynamic_bsz=True, offload=True
lr=1e-6, lr_warmup=10, weight_decay=0.1
staleness_threshold=0.5 (exp2/3/4)
trigger_parameter_sync_step=4 (exp2/3/4)
```

---

## 四组实验

### Exp1 — Colocate 基线

**脚本**：[exp1_colocate_baseline.sh](exp1_colocate_baseline.sh)

**架构**：同步 PPO，rollout 与 training 共享同一批 GPU（colocate 模式）

```
4 nodes × 8 GPUs = 16 GPUs（共享）
入口：verl.trainer.main_ppo
rollout.mode=native
```

**目的**：建立传统同步训练的性能与效果基线。

---

### Exp2 — Fully Async + 不重算

**脚本**：[exp2_fully_async_no_recompute.sh](exp2_fully_async_no_recompute.sh)

**架构**：rollout 与 training 完全解耦，`old_log_prob` 直接复用 vLLM 生成时的值

```
Rollout: 2 nodes × 8 GPUs = 16 GPUs
Trainer: 2 nodes × 8 GPUs = 16 GPUs
总计: 32 GPUs（与 Exp1 对齐）

bypass_mode=True
enable_standalone=False
```

**目的**：fully_async 吞吐基线；验证与 Exp1 相比的加速比。

**关注指标**：`fully_async/count/staleness_samples`（stale 样本比例）

---

### Exp3 — Fully Async + CPU 重算

**脚本**：[exp3_fully_async_cpu_recompute.sh](exp3_fully_async_cpu_recompute.sh)

**架构**：rollout 与 training 解耦，`old_log_prob` 由 trainer GPU 串行重算（旧参数存于 CPU RAM）

```
Rollout: 2 nodes × 8 GPUs = 16 GPUs
Trainer: 2 nodes × 8 GPUs = 16 GPUs（重算占用额外时间）

bypass_mode=False
enable_standalone=False
```

**目的**：验证重算的正确性收益是否值得付出串行开销；与 Exp2 对比效果差异。

**关注指标**：`old_log_prob_mfu`、CPU↔GPU 搬运耗时、训练稳定性

---

### Exp4 — Fully Async + ModelEngineServer 重算

**脚本**：[exp4_fully_async_model_engine_server.sh](exp4_fully_async_model_engine_server.sh)

**架构**：独立专用 GPU 节点异步计算 `old_log_prob`，与 training 完全并行，trainer 零等待

```
Rollout:           2 nodes × 8 GPUs = 16 GPUs
Trainer:           2 nodes × 8 GPUs = 16 GPUs
ModelEngineServer: 1 node  × 8 GPUs =  8 GPUs（TP=1, PP=1, forward-only）
总计: 40 GPUs（+8 vs Exp2/3）

bypass_mode=False
enable_standalone=True
batch_size=64, timeout=5s
```

**目的**：验证 model_engine_server 能否消除 Exp3 的串行重算开销；评估额外 8 GPUs 的投入产出比。

**关注指标**：server GPU 利用率、batch timeout 触发比例、与 Exp3 的吞吐对比

---

## 对比矩阵

| | Exp1 Colocate | Exp2 Async+不重算 | Exp3 Async+CPU重算 | Exp4 Async+Server重算 |
|---|:---:|:---:|:---:|:---:|
| **GPU 总数** | 16（共享） | 32 | 32 | 40 |
| **Rollout/Training 解耦** | 否 | 是 | 是 | 是 |
| `bypass_mode` | N/A | True | False | False |
| `enable_standalone` | N/A | False | False | True |
| **old_log_prob 来源** | vLLM（同步） | vLLM rollout_log_probs | Trainer GPU（串行） | ModelEngineServer（并行） |
| **额外资源** | — | — | ~14 GB CPU RAM/node | 8 GPUs |
| **算法正确性** | 标准 PPO | 近似（bypass） | 精确 | 精确 |

---

## 核心对比问题

1. **Exp1 vs Exp2**：fully_async 在等量 GPU 下带来多少吞吐加速？
2. **Exp2 vs Exp3**：bypass 与精确重算的效果差距有多大？重算是否必要？
3. **Exp3 vs Exp4**：model_engine_server 能否消除串行重算瓶颈？额外 8 GPUs 值不值？
4. **Exp2 vs Exp4**：在效果相近的前提下，server 方案的吞吐上限是多少？

---

## 文件列表

```
exp/
├── README.md                               # 本文档
├── runtime_env.yaml                        # Ray 公共环境变量
├── exp1_colocate_baseline.sh
├── exp2_fully_async_no_recompute.sh
├── exp3_fully_async_cpu_recompute.sh
└── exp4_fully_async_model_engine_server.sh
```

提交 Ray job 时通过 `--env TENSORBOARD_DIR=<path>` 按实验名单独指定 tensorboard 目录。
