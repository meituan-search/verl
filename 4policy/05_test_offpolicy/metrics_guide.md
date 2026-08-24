# staleness_sweep Trainer — Metrics Viewing Guide

> 实验性 V1 PPO trainer mode,用于诊断 off-policy 效应。
> 通过一个 cycle 内 N 步的 staleness 扫描,观察 0..N-1 step staleness 下三个 log_prob 的 diff。

## 1. 设计回顾

**Cycle 逻辑**:每 N 步一个 cycle。Cycle 开始时一次性 rollout `N * train_batch` 样本(用 W_0 解码),随后 N 个 train step 各消费一个 train_batch。N 步跑完后,用更新后的权重开启下一个 cycle。

**每步三个 log_prob**:

| log_prob | 引擎 | 权重 | 含义 |
|---|---|---|---|
| `rollout_log_probs` | rollout | W_0(解码时) | 一次性 rollout 时记录,固定不变 |
| `new_rollout_log_probs` | rollout | W_{k-1}(当前) | `on_sampled` 中 reprefill 得到,随 k 漂移 |
| `old_log_probs` | train | W_{k-1}(当前) | `_compute_old_log_prob` 在 train 引擎上算 |

其中 k = 1..N 是 cycle 内的 step 序号。rollout 引擎在每步 `on_step_end` sync,因此 step k 的 rollout 引擎权重为 W_{k-1}。

## 2. 三组 log_prob diff

`compute_offpolicy_metrics`(在 `verl/trainer/ppo/rollout_corr_helper.py`)在收到 `new_rollout_log_prob` 参数后,自动分解为三组 ratio:

| 组 | ratio 公式 | 解读 |
|---|---|---|
| `staleness/*` | `new_rollout - rollout` = log(π_{W_{k-1}} / π_{W_0}) | 纯权重 staleness — (k-1) 步更新后,同一条 trajectory 在新权重下的 log-prob 漂移 |
| `mismatch/*` | `old - new_rollout` = log(π_old^{train} / π_new-rollout^{rollout}) | T/R 引擎同权重下的数值差 — train 引擎 vs rollout 引擎在 W_{k-1} 的实现差异 |
| 综合(无前缀) | `old - rollout` = log(π_{W_{k-1}} / π_{W_0}) | 综合 off-policy gap(前两者之和),即 PPO 实际使用的 ratio |

恒等式:`staleness + mismatch == combined`(代码中有 1e-5 容差检查,违反会 warning)。

> **前缀包装**:`compute_offpolicy_metrics` 是底层函数,返回无前缀综合组 + `staleness/*`、`mismatch/*` 新组。两条调用路径分别在外层加不同前缀:
> - IS/RS 路径:`compute_rollout_correction_and_rejection_mask`(rollout_corr_helper.py:917-923)加 `rollout_corr/` 前缀 → `rollout_corr/kl`、`rollout_corr/staleness/kl` 等
> - staleness_sweep 路径:`_compute_staleness_metrics`(trainer_staleness_sweep.py 末尾)加 `offpolicy/` 前缀 → `offpolicy/kl`、`offpolicy/staleness/kl` 等
>
> 下文示例均用 staleness_sweep 路径的 `offpolicy/` 前缀。`staleness_sweep/*` 是 trainer 自身 meta,不包装。

## 3. staleness sweep 如何产生 0..N-1

Cycle 内 N 个连续 global_step,rollout 引擎权重从 W_0 单调推进到 W_{N-1}:

| cycle 内 step | rollout 引擎权重 | staleness/* 测量 | 对应 staleness |
|---|---|---|---|
| k=1 | W_0 | log(π_{W_0} / π_{W_0}) | **0 step staleness** |
| k=2 | W_1 | log(π_{W_1} / π_{W_0}) | **1 step staleness** |
| k=3 | W_2 | log(π_{W_2} / π_{W_0}) | **2 step staleness** |
| ... | ... | ... | ... |
| k=N | W_{N-1} | log(π_{N-1} / π_{W_0}) | **(N-1) step staleness** |

Cycle 边界:第 N+1 步时 `_steps_since_rollout` 归零,提交下一批 N*train_batch,rollout 引擎重新对齐 W_0(此时的 W_0 是上一 cycle 结束时的最终权重),staleness 重新从 0 开始。

## 4. 输出的 metric 名

### 4.1 `offpolicy/staleness/*` 与 `offpolicy/mismatch/*`(各 17 个)

由 `for prefix, lr in (("staleness", log_ratio_stale), ("mismatch", log_ratio_mis)):` 循环产出,经外层包装后实际 metric 名为 `offpolicy/{prefix}/...`。每组:

| metric | 说明 |
|---|---|
| `offpolicy/{prefix}/kl` | 直接 KL 估计:-E[log_ratio](r = π_p/π_q) |
| `offpolicy/{prefix}/k3_kl` | K3 KL 估计:E[exp(lr) - lr - 1] |
| `offpolicy/{prefix}/chi2_token` | token 级 χ²:E[ρ²] - 1 |
| `offpolicy/{prefix}/chi2_seq` | seq 级 χ²:E[(Πρ_t)²] - 1 |
| `offpolicy/{prefix}/mean` | 符号均值 — 正值 = 新权重低估 |
| `offpolicy/{prefix}/abs_mean` | L1 幅值 — 抗符号抵消 |
| `offpolicy/{prefix}/std` | token 间 std |
| `offpolicy/{prefix}/min`, `offpolicy/{prefix}/max` | 极值(检测 outlier token) |
| `offpolicy/{prefix}/fraction_high` | \|log_ratio\| > 0.5 的 token 占比 |
| `offpolicy/{prefix}/fraction_low` | \|log_ratio\| < 0.1 的 token 占比 |
| `offpolicy/{prefix}/positive_fraction`, `offpolicy/{prefix}/negative_fraction` | 符号分布 |
| `offpolicy/{prefix}/seq_mean`, `offpolicy/{prefix}/seq_abs_max`, `offpolicy/{prefix}/seq_std` | per-seq 统计 |
| `offpolicy/{prefix}/eff_sample_size`, `offpolicy/{prefix}/eff_sample_size_ratio` | Kish's ESS |

### 4.2 综合组(`offpolicy/` 前缀,13 个)

`compute_offpolicy_metrics` 产出的综合组(rollout vs old),经外层包装后实际 metric 名为 `offpolicy/...`:

`offpolicy/kl`, `offpolicy/k3_kl`, `offpolicy/chi2_token`, `offpolicy/chi2_seq`, `offpolicy/ppl_ratio`, `offpolicy/training_ppl`, `offpolicy/training_log_ppl`, `offpolicy/rollout_ppl`, `offpolicy/rollout_log_ppl`, `offpolicy/log_ppl_diff`, `offpolicy/log_ppl_abs_diff`, `offpolicy/log_ppl_diff_max`, `offpolicy/log_ppl_diff_min`。

### 4.3 `training/*` 三两两 pairwise diff(由 `calculate_debug_metrics` 产出)

`verl/utils/debug/metrics.py` 的 `calculate_debug_metrics` 在 `_compute_old_log_prob` 中被调用,基于三个 log_prob 做两两 pairwise diff。当 `new_rollout_log_probs` 在 `data.batch` 中时(staleness_sweep 通过 `_debug_log_prob_extra_fields` 钩子注入),额外产出两组:

| 组 | 对比 | metric |
|---|---|---|
| Pair 1(combined,原有) | `rollout_log_probs` vs `old_log_probs` | `training/rollout_probs_diff_{valid,max,mean,std}`,`training/rollout_actor_probs_pearson_corr` |
| Pair 2(staleness,新增) | `new_rollout_log_probs` vs `rollout_log_probs` | `training/new_rollout_vs_rollout_probs_diff_{valid,max,mean,std}`,`training/new_rollout_rollout_probs_pearson_corr` |
| Pair 3(mismatch,新增) | `old_log_probs` vs `new_rollout_log_probs` | `training/old_vs_new_rollout_probs_diff_{valid,max,mean,std}`,`training/old_new_rollout_probs_pearson_corr` |

每对包含:`valid`(0/1)、`max`、`mean`、`std` 的 abs-logprob-diff,加一个 Pearson 相关系数。diff 用 `|log_probs1 - log_probs2|` 在 masked token 上计算;Pearson 用 `torch.corrcoef` 在 masked token 上计算。

> 与 §2 的 `compute_offpolicy_metrics` 区别:那边产出的是 KL/χ²/ESS 等 ratio-based 统计(`staleness/*`、`mismatch/*`、无前缀综合组);这边产出的是 pairwise abs-diff + Pearson,语义更直白,适合快速 sanity check。

### 4.4 `staleness_sweep/*`(trainer 自身 meta)

| metric | 写入位置 | 说明 |
|---|---|---|
| `staleness_sweep/resume_version` | `on_sampled` | reprefill 用的权重版本号(= `global_steps - 1`) |
| `staleness_sweep/steps_since_rollout` | `on_sampled` | 距 cycle 开始的步数倒计数(N-1, N-2, ..., 0) |
| `staleness_sweep/sample_staleness_mean` | `_compute_staleness_metrics` | 本 step 所有 sample 的平均 staleness(应 = k-1) |
| `staleness_sweep/sample_staleness_max` | `_compute_staleness_metrics` | 最大 staleness |

## 5. 如何查看

### 5.1 Wandb(推荐)

`ppo_trainer.yaml` 默认 `logger: ["console", "wandb"]`。打开 wandb run 页面:

1. **验证 sweep 正确性**:画 `staleness_sweep/sample_staleness_mean` vs `_step` — 应该看到锯齿波,N 步一个 cycle 从 0 爬到 N-1 然后跳回 0。
2. **观察 staleness 增长**:画 `offpolicy/staleness/abs_mean` vs `_step` — 应该看到 cycle 内单调递增(权重更新越多,新权重 vs 旧 rollout 的 diff 越大)。
3. **对比三组**:把 `offpolicy/staleness/abs_mean`、`offpolicy/mismatch/abs_mean`、`offpolicy/abs_mean`(综合组无对应 abs_mean,可看 `offpolicy/log_ppl_abs_diff`)叠加 — 验证 `staleness + mismatch ≈ combined`。
4. **关键 KL 曲线**:`offpolicy/staleness/kl` 和 `offpolicy/staleness/k3_kl` — 看 KL 随 staleness 增长趋势。
5. **ESS 衰减**:`offpolicy/staleness/eff_sample_size_ratio` — staleness 增大时有效样本量应衰减。

搜索 prefix:wandb 左侧 metric 树里搜索 `offpolicy/staleness/`、`offpolicy/mismatch/`、`offpolicy/`(综合组)、`staleness_sweep/`、`training/`。

### 5.2 Console

每个 global_step 的 logger 输出包含上述所有 metrics。在训练日志里 grep:

```bash
# 跟踪一个 cycle 内 staleness 增长
grep "staleness_sweep/sample_staleness_mean" train.log

# 看 staleness KL
grep "offpolicy/staleness/kl" train.log

# 看 mismatch(应相对稳定,与 staleness 无关)
grep "offpolicy/mismatch/abs_mean" train.log

# 看 training/* pairwise diff(Pair 1/2/3)
grep "training/rollout_probs_diff_mean\|training/new_rollout_vs_rollout_probs_diff_mean\|training/old_vs_new_rollout_probs_diff_mean" train.log
```

### 5.3 离线分析

`metrics` dict 在每个 step 都会包含上述所有键。如果接了自定义 logger(如 tensorboard),只需按 prefix 过滤即可。

## 6. 验证 staleness sweep 是否正确

跑一个 `num_steps=8` 的 cycle,检查前 8 个 global_step:

| global_step | 期望 `sample_staleness_mean` | 期望 `offpolicy/staleness/abs_mean` |
|---|---|---|
| G+1 | 0 | ≈ 0 |
| G+2 | 1 | 小幅增长 |
| G+3 | 2 | 继续增长 |
| ... | ... | ... |
| G+8 | 7 | 最大 |

- 如果 `sample_staleness_mean` 不是 0,1,2,...,7 → cycle 逻辑或权重 sync 时机有问题
- 如果 `offpolicy/staleness/abs_mean` 在 step 1 不是 ≈ 0 → reprefill 或 log_prob 切片有 bug
- 如果 `offpolicy/staleness/abs_mean + offpolicy/mismatch/abs_mean` 与综合组(`offpolicy/log_ppl_abs_diff` 等)严重不一致 → 三组 log_prob 不对齐(可能是 token 顺序/padding 问题)

## 7. 启用方式

```bash
trainer.v1.trainer_mode=staleness_sweep \
trainer.v1.staleness_sweep.num_steps=8
```

依赖的 plumbing(均已 stage 到 `dev/wsl_v1_staleness_dev0` 分支):

- `verl/trainer/ppo/v1/trainer_staleness_sweep.py` — trainer 主体;`_compute_staleness_metrics` 给 `compute_offpolicy_metrics` 返回值加 `offpolicy/` 前缀(staleness_sweep 路径专用,与 IS/RS 路径的 `rollout_corr/` 区分)
- `verl/trainer/ppo/v1/__init__.py` — 注册
- `verl/trainer/config/ppo_trainer.yaml` — `staleness_sweep.num_steps` config
- `verl/trainer/ppo/v1/trainer_base.py` — `on_sampled` 钩子(父类在 sample 后调用)+ `_debug_log_prob_extra_fields` 钩子(把 `new_rollout_log_probs` 带入 `calculate_debug_metrics` 的 data 中)
- `verl/trainer/ppo/rollout_corr_helper.py` — `compute_offpolicy_metrics` 的 `new_rollout_log_prob` 参数 + staleness/mismatch 分解;`compute_rollout_correction_and_rejection_mask` 的外层 `rollout_corr/` 包装(IS/RS 路径,与 staleness_sweep 的 `offpolicy/` 区分)
- `verl/utils/debug/metrics.py` — `calculate_debug_metrics` 扩展:当 `new_rollout_log_probs` 在 `data.batch` 中时,产出 Pair 2/3 pairwise diff

不依赖 4policy trainer(`trainer_colocate_async_4policy.py`)或 `algorithm.py` 的 4policy config 字段。
