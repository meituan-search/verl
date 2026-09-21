# Partial Reprefill Trainer 优化设计

> 适用于 `dev/wsl_v1_staleness_dev0` 分支上的 `partial_reprefill` trainer 模式。
> 目标：在现有 `num_warmup_batches` 固定回放缓冲深度的基础上，进一步优化为
> 动态调度，使 KV 利用率、staleness、group 内聚性三者协同。

## 1. 优化目标与优先级

三个目标按优先级从高到低排列，高优先级目标在冲突时优先满足：

### P0：KV 长期占满（SGLang 职责，非 verl 调度目标）
- **语义**：rollout 引擎的 KV cache 空间长期保持高占用，是 **SGLang**
  内置调度策略的职责，不是 verl 侧需要量化或保证的目标。
- **verl 侧的职责**：每 step 把该 step 新增的请求全部交给 SGLang，
  持续供请求即可，不干预 KV 层面调度。
- **若 KV 占用率不达标**：说明 batch_size 相对偏小，是训练人需要调
  `batch_size` 或 `num_warmup_batches` 的超参问题。
- **若 KV 不够用**：SGLang 内置策略选择一部分请求执行、一部分在内部
  queue 排队，由 SGLang 自行处理。

### P1（软目标，无上界）：staleness 最小化
- **语义**：trainer 消费到的样本 staleness 尽量小，无上界约束。
- **度量粒度**：token 粒度（越精细越好），对应
  `offpolicy_token/staleness_mean`（见 `reprefill_utils.py` 中
  `compute_and_emit_token_staleness_metrics`）。
- **含义**：在 P0 满足的前提下，尽量让 trainer 用到的数据 staleness 小；
  不再使用固定 `num_warmup_batches` 作为 staleness 上界。
- **为何不设上界**：staleness 大小取决于 rollout 与 trainer 的相对速度，
  硬上界会迫使系统在 rollout 快于 trainer 时人为制造 staleness，或在
  rollout 慢时无法达到上界——应让 staleness 自然形成。

### P2（软目标，SGLang 负责实现）：group 内聚性
- **语义**：在 `partial_rollout` 多 segment 语义下，单条 trajectory 跨
  多个权重版本是固有特性（abort 后用新权重继续生成剩余 token）。因此
  P2 的真实目标是：**group 内 sibling 的 segment 集合尽量一致**——即
  sibling 的被打散节奏趋同（都在 V1+V2，而不是 A1 在 V1+V2、A2 在
  V2+V3），而非"绝对不跨版本"。
- **机制**：SGLang 通过 retract/admission affinity 集中打散/准入同一
  group 的 sibling，让 sibling 的被打散节奏一致 → segment 集合趋同 →
  group 内版本分布更均匀。
- **约束级别**：软目标，不强制整个 group 原子地同时运行/停止。
- **负责人**：SGLang 侧实现；trainer 侧只负责传递 `group_id`。

## 2. Staleness 度量

- **粒度**：token 级，使用 `token_versions` 字段（见 `agent_loop_tq.py`）
  与 `current_version = global_steps - 1` 之差。
- **统计指标**：
  - `offpolicy_token/staleness_mean`：staleness>0 的 token 上
    `|log π_rollout - log π_new_rollout|` 均值。
  - `offpolicy_token/fresh_token_ratio`：gap==0 的 token 占比。
  - `offpolicy_token/stale_token_ratio`：gap>0 的 token 占比。
  - `offpolicy_token/staleness_by_version_gap_{0,1,2_3,4plus}`：版本差
    分桶占比。
- **上界**：不设置具体值。

## 3. Group Affinity Scheduling（P2 实现机制）

Group 并不强制整个 group 同时运行/停止。当 KV 不足需要停止部分在飞
request，或 KV 有空需要 admit 等待中的 request 时，按以下 affinity
规则选择：

### 3.1 Retract Affinity（KV 不足时驱逐）
- **规则**：KV 不足需要 retract 某个/某几个在飞 request 时，优先 retract
  已经有 sibling 被 retract 的 group 中的在飞 request。
- **直觉**：集中 retract 同一 group，使该 group 整体进入"部分被驱逐"状态，
  避免"打散驱逐"导致多个 group 都进入半完成状态、group 内聚性被破坏。

### 3.2 Admission Affinity（KV 有空时准入）
- **规则**：KV 有空需要 admit 等待中的 request 时，优先 admit 已经有
  sibling 在飞的 group 中的等待 request。
- **直觉**：集中 admit 同一 group 的剩余 sibling，使该 group 整体尽快完成，
  减少 group 内 sibling 跨版本的概率。

### 3.3 Tie-breaker
- **规则**：当 affinity 规则无法唯一决定（多个候选 group 同样满足
  affinity 条件），沿用 SGLang 现有的 retract/admission tie-breaker
  策略，不引入新的 tie-breaker。
- **含义**：本次优化只在现有策略前加一层 group affinity 预排序，不改动
  现有策略本身。

### 3.4 Group Affinity 与 "卡死" 的关系
- **软 affinity 不死锁**：affinity 规则只影响排序偏好，不是硬约束。极端
  情况下（如所有在飞 request 都属于无 retract sibling 的 group、或所有
  等待 request 都属于无在飞 sibling 的 group），affinity 退化，自然
  fall through 到 SGLang 现有策略，不会出现死锁或饥饿。
- **不需要额外的 anti-starvation 机制**：依赖 SGLang 现有策略的
  公平性保证。

## 4. Retract 与 Abort 的独立性

- **Retract**：发生在 rollout 过程中，由 KV 不足触发；被 retract 的
  request 在 KV 空间腾出后可继续（resume）。
- **Abort**：由权重更新触发，终止所有正在运行的 request；与 retract 无关。
- **二者独立**：retract 不导致 request 终止；abort 不区分 request 是否曾被
  retract。

## 5. group_id 传递链路

```
agent_loop → rollout client → SGLang GenerateReqInput
```

- **agent_loop 侧**：为每个 `rollout.n` group 内的 sibling 轨迹打上相同
  `group_id`（基于 uid 生成）。
- **rollout client 侧**：透传 `group_id` 到 SGLang 的 `GenerateReqInput`
  参数。
- **SGLang 侧**：scheduler 读取 `group_id`，用于 §3 的 affinity 预排序。

## 6. 责任划分

### Trainer 侧
- **简单消费**：判断 TQ 是否已生产够一个 batch 的数据，够了即消费，不够则等待。
  不做 staleness-aware 排序、不做 group 选择——trainer 侧不引入额外调度逻辑。
- 所有 staleness / group 优化均由 SGLang 侧负责。

### SGLang 侧
- **group-affinity 预排序**：在现有 retract/admission 候选选择策略前，
  按 §3 规则对候选 request 做基于 `group_id` 的 affinity 排序。
- **tie-breaker**：沿用现有策略。
- **staleness 控制**：通过 retract/admission 调度自然形成 staleness，
  不在 trainer 侧做任何过滤或排序。

## 7. 与现有实现的关系

- 现有 `num_warmup_batches` 固定回放深度作为 v0 baseline 保留。
- 现有 `staleness_sweep` 模式的 token staleness 指标
  (`offpolicy_token/*`) 直接复用，无需新增 metric。
- 现有 SGLang prometheus endpoint（port 44398）可观测 KV 利用率，
  作为 P0 的稳态监控。

## 8. 后续实施 TODO

1. SGLang 侧：retract/admission 候选的 group-affinity 预排序实现。
2. 传递链路：`group_id` 从 agent_loop 到 `GenerateReqInput` 的透传。
3. Trainer 侧：确认消费逻辑保持"够一个 batch 即消费"的简单语义，
   不引入额外调度。
