# Partial Reprefill Trainer 优化设计 — 待确定问题记录

> 与设计文档 `docs/partial_reprefill_optimization_design.md` 配套。
> 通过对话逐个解决下列问题，每项解决后填入"决议"字段，并更新主设计文档。

---

## Q1: P0 KV 占满量化判据

**问题**：P0 "长期占满"没有量化阈值与统计窗口，无法判断是否达成。

**决议**：**不需要在 verl 侧量化 P0。**

理由：
- KV 占满是 SGLang 的责任，由 SGLang 内置调度策略处理。
- verl 每个 step 向 SGLang 提交的请求数是确定的——每 step 把该 step 新增的
  请求全部交给 SGLang，verl 不做 KV 层面的调度决策。
- 若请求量达不到高 KV 占用率，说明 batch_size 相对偏小，是**训练人**
  需要调 `batch_size` 或 `num_warmup_batches` 的超参问题，不是框架要管的。
- 若 KV 不够用，SGLang 根据内置策略选择一部分请求执行、一部分在 SGLang
  内部 queue 排队——这也是 SGLang 的职责，verl 不干预。

P0 在 verl 侧的实际语义简化为：**verl 持续向 SGLang 供请求**（每 step 一个
batch 的请求量），不主动调控 KV 占用。

---

## Q2: P0 硬/软语义自相矛盾

**问题**：§1 写 "稳态软目标（steady-state soft target）"，又说 "约束级别：硬目标"。
P0 到底是硬目标还是软目标？这决定 retract/admission 中 KV 占满与其他规则冲突时的优先级。

**决议**：**Q1 的决议让此问题消解。**

P0 在 verl 侧根本不是调度目标，所以硬/软之分不再适用。主设计文档 §1 的
P0 描述需要改写——P0 从"verl 的调度硬/软目标"改为"SGLang 的固有职责，
verl 只需持续供请求"。

P1/P2 与 P0 之间在 verl 侧**不存在冲突需要谁让步**——P0 不是 verl 的
调度维度。

---

## Q3: P1 staleness 收敛判据

**问题**：staleness 无上界 OK，但 "尽量小" 是相对什么？需要对比基准。

**决议**：**不需要在 verl 侧设收敛判据。**

理由：
- 与 Q1 同构：staleness 是 SGLang 内在调度机制的自然结果，不是 verl 要
  量化或保证的目标。
- "staleness 尽量小" 由 group affinity 调度机制内在保证：
  - 优先让资源以 group 粒度尽量完成，不要很多 group 都只完成一部分。
  - group 一旦开始生产就尽量一次性完成 → sibling 不跨版本 → staleness 小。
- verl 侧 `offpolicy_token/*` metric 只作为**诊断信号**（看趋势 + 异常告警），
  不作为收敛判据。
- 没有 v0 baseline 对比需求，也没有"必须收敛到某值"的硬指标。

---

## Q4: "够一个 batch"的判据（关键）

**问题**：trainer 消费的"够"是：
- (A) 够 `batch_size` 条 trajectory（任意 group 拼凑）
- (B) 够 `batch_size / rollout.n` 个**完整 group**（必须整组到齐）

选 (A) 则半截 group 可能进训练；选 (B) 则 group 内聚性责任偷偷挪回 trainer 侧。

**待定**：
- A or B？

**决议**：**(B) — 必须整组到齐。**

理由：
- GRPO 算法的硬性要求：group 内 sibling 一起计算 advantage，只完成一部分
  group 是无法进行 RL 训练的。
- 这不是"调度选择"，而是 GRPO 算法的**消费契约**。
- 与 Q1/Q3"verl 侧不做调度"的原则不冲突——group 粒度消费是算法契约的
  体现，不是 verl 主动的 KV/staleness 调度决策。
- "够一个 batch"的精确含义：TQ 里有 `batch_size / rollout.n` 个完整 group
  到齐，trainer 才消费；否则继续等待。

---

## Q5: group_id 唯一性范围

**问题**：group_id 是全局唯一、batch 内唯一还是跨 batch 复用？
若跨 batch 复用，不同 batch 的 sibling 会被当成同一 group 互相影响 retract/admission。

**待定**：
- 唯一性范围

**决议**：
- **唯一性范围**：全局唯一（整个训练 run 内每个 group 的 group_id 不重复）。
- **生成方式**：确定性组合 `f"{step}_{prompt_idx}"`。
  - `step` = trainer 的 global_steps（生产该 group 时的 step 号）。
  - `prompt_idx` = 该 group 在其 step 内的 prompt 索引。
  - 可调试性好——从 group_id 能反推出 step / prompt。
  - 不需要维护"已退役 group_id"集合：SGLang 只看当前在飞/等待中的
    request 的 group_id，group 一旦完成其 group_id 自然不再活跃。

---

## Q6: group_id 来源字段

**问题**："基于 uid 生成"，但 uid 指 prompt uid 还是 group uid？字段名是什么？
现有代码里哪个字段对应？

**待定**：
- 来源字段

**决议**：
- **生成方式**：`group_id = f"{step}_{uid}"`。
  - `step` ← `trajectory["step"]`（即 `kwargs["global_steps"]`）。
  - `uid` ← `prompt["uid"]`（dataset 原生 prompt 唯一标识，全局唯一）。
  - 加 `step` 前缀解决跨 step 重复（同 prompt 被多 step 采样时区分）。
- **生成位置**：`_run_prompt`（`agent_loop_tq.py:118`）。
  - 此处 `uid` 与 `trajectory["step"]` 都在 scope 内。
  - group_id 在 group 生命周期起点就确定，传给 n 个 sibling 共享。
- **现有 id 字段对照**：
  - `uid`（dataset 原生）= group 的天然标识，n 条 sibling 共享。
  - `session_id` ∈ [0, n) = group 内 sibling index。
  - TQ key `f"{uid}_{session_id}_{i}"` = trajectory 级唯一（i 是 agent loop 多输出 index）。
  - group_id 不替代 TQ key，只是新增的一个字段，随 sibling request 传给 SGLang。

---

## Q7: rollout client 指哪一层

**问题**：group_id 注入点是：
- `LLMServerClient.generate`（直接 remote call）
- `FullyAsyncLLMServerClient.generate`（multi-segment retry loop）
- 还是两者都要？

**待定**：
- 注入点

**决议**：
- **逻辑注入点**：(A) 基类 `LLMServerClient.generate`（`llm_server.py:230`）。
  - group_id 是 request 级属性，不是 multi-segment 重试逻辑专属的。
  - `server.generate.remote(...)` 调用在基类（line 274-283），group_id 从这里传给
    SGLang `GenerateReqInput`。
  - `FullyAsyncLLMServerClient.generate` 内部 `super().generate()`（line 501）
    会自动获得 group_id 透传能力。
- **签名改动**（参数 plumbing 必经两层）：
  - 基类 `LLMServerClient.generate`：加 `group_id: str | None = None` 关键字参数。
  - 派生类 `FullyAsyncLLMServerClient.generate`：加同参数，透传给内部 `super().generate()`。
  - 这不是冗余——派生类只做透传，逻辑注入只在基类。
- **上游 caller plumbing**：
  - `agent_loop` 侧调用 `client.generate(...)` 时显式传 `group_id`。
  - group_id 在 `_run_prompt` 生成（Q6 决议），通过 kwargs 传到 `_run_agent_loop`，
    再到 `client.generate` 调用点。
- **server 侧**（`async_sglang_server.py` 的 `generate`）：
  - 接收 group_id，塞进 `GenerateReqInput`，供 SGLang scheduler 使用。

---

## Q8: 首次 retract 选择规则

**问题**：没有任何 group 有 sibling 被 retract 时，第一次 retract 的候选选择规则。
此时系统大部分时间处于"首次"状态（稳态前）。

**待定**：
- 选择规则（随机/LIFO/最长在飞时间?）

**决议**：
- **SGLang 现有 retract 机制**（`schedule_batch.py:2860-2921`）：
  - `retract_decode` 是**逐个 retract** 的循环——每次 pop 一个 request，
    立即 `check_decode_mem`，不够再 pop 下一个。
  - 默认 retract 顺序由 `_get_decode_retraction_order`（line 2924）决定。
- **默认 retract 策略**（`length_key`，line 2934）：
  - 主 key：`len(req.output_ids)`（已生成 output token 数）。
  - `reverse=True`：output 短的排后面，先被 retract。
  - **语义**：优先 retract output 短的 request（KV 累积少，损失小）；
    保留 output 长的（快完成，避免前功尽弃）。
  - 和候选 (B)/(D) "retract 新 request" 方向一致但更精细。
- **Q8 决议**：(A) 沿用 SGLang 现有 retract 策略（`length_key`）作为
  affinity 退化时的 fallback。
- **group affinity 作为动态预排序层**（实现方式 (Y) 动态重评估）：
  - 每次 pop 前按"当前是否有 sibling 被 retract"重新评估候选。
  - retract 第 1 个：affinity 退化 → fall through 到 `length_key`。
  - retract 第 2 个起：优先从已有 sibling 被 retract 的 group 中选，
    实现 §3.1 "集中 retract 同一 group" 语义。
- **叠加顺序**：(i) group affinity 为主 key，`length_key` 为 tie-breaker。
  - 符合 §3.3"tie-breaker 沿用 SGLang 现有策略"。
- **priority 策略**：默认走 `length_key`，不启用 priority。
- **改动范围**：会改 SGLang `retract_decode` 循环结构（从静态预排序改为
  动态重评估），改动量可接受。

---

## Q9: 首次 admit 选择规则

**问题**：没有任何 group 有 sibling 在飞时，第一次 admit 的候选选择规则。

**待定**：
- 选择规则

**决议**：
- **SGLang 现有 admit 机制**（`scheduler.py:3231` `get_new_batch_prefill`）：
  - `waiting_queue` 是 FIFO 列表（`List[Req]`，line 1151）。
  - admit 时按 FIFO 顺序遍历 `waiting_queue`，每个 req 用
    `adder.add_one_req` 尝试加入 prefill batch。
  - `calc_priority`（line 3307）在遍历前排序——启用 priority 时按 priority
    排序，否则默认 FIFO。
  - admit 是**批量**的——一次循环把能加的都加进 `can_run_list`，
    直到 `batch_is_full` 或 `NO_TOKEN`。
- **默认 admit 策略**：**FIFO**（先来先服务）。
- **Q9 决议**（同构 Q8）：
  - (A) 沿用 SGLang 现有 admit 策略（FIFO）作为 affinity 退化时的 fallback。
  - group affinity 作为**动态预排序层**（实现方式 (Y) 动态重评估）：
    - 每次 admit 一个 req 前按"当前是否有 sibling 在飞"重新评估候选。
    - admit 第 1 个：affinity 退化 → fall through 到 FIFO。
    - admit 第 2 个起：优先 admit 已有 sibling 在飞的 group 的剩余 request，
      实现 §3.2 "集中 admit 同一 group" 语义。
- **叠加顺序**：group affinity 为主 key，FIFO 为 tie-breaker
  （符合 §3.3）。
- **priority 策略**：默认 FIFO，不启用 priority。
- **改动范围**：会改 SGLang admit 循环结构（从静态 FIFO 遍历改为动态
  重评估），改动量可接受。

---

## Q10: resume 时 group_id 是否保持

**问题**：被 retract 的 request resume 时 group_id 不变吗？
若不变，retract affinity（§3.1）的判断 "已经有 sibling 被 retract" 是否把 resume 后的 sibling 也算进去？

**待定**：
- group_id 是否在 resume 时保持
- retract sibling 判定范围

**决议**：
- **问题 1：resume 时 group_id 保持**。
  - SGLang `retract_decode`（`schedule_batch.py:2860`）只把 request 从
    running batch 移到 `waiting_queue`（`_add_request_to_queue`，line 2791），
    **request 对象不重建**。
  - group_id 作为 request 的属性字段，retract/resume 过程中自然保持，
    不需要额外操作。
- **问题 2：retract affinity 判断范围选 (I)** —— 只算当前在 waiting_queue
  的 sibling。
  - "已经有 sibling 被 retract" = 当前有 sibling 在 waiting_queue
    （被 retract 但未 resume）。
  - 不算"曾被 retract 但已 resume 回 running batch"的历史标记。
- **问题 3：(I) 自然避免饥饿**。
  - (II)（算历史标记）会导致 group 一旦被 retract 过就永远带标记，
    后续 retract 永远优先选该 group → 反复被打散、整组无法完成 → 饥饿。
  - (I) 只看当前状态：group 恢复完整后 affinity 消失，下次 retract 重新
    评估，没有永久标记 → 自然避免饥饿。
  - 符合 §3.4 "软 affinity 不死锁"原则。

---

## Q11: abort 与 group 的交互

**问题**：权重更新 abort 所有在飞 request 时，等待中/半截 group 的 sibling 怎么办？
继续等下一个权重版本 resume？还是一起 abort？
这影响下一版本 P2 affinity 的初始状态。

**待定**：
- abort 时等待中 sibling 的处置

**决议**：**问题消解——partial_rollout 多 segment 机制天然处理。**

理由：
- SGLang `abort_all=True` 现有行为（`scheduler.py:4611`）：清空所有状态
  （waiting_queue / running_batch / retracted_queue / disagg 队列），
  KV 释放，权重更新。
- **verl 侧不丢弃 group、不重生**——`FullyAsyncLLMServerClient.generate`
  （`llm_server.py:499`）的 multi-segment while 循环继续工作：
  - abort 后，`super().generate(prompt_ids + final_output.token_ids, ...)`
    把已生成部分作为新 prompt 重新发给 SGLang。
  - SGLang 用**新权重**继续生成剩余 token。
  - 对 verl 侧来说，这个 request 没有失败——只是跨多个 segment 完成。
- 同一 group 的 n 条 sibling 都走这个路径，group 完整性保持；
  **group_id 在 resume 时保持**（Q10 决议，request 对象不重建）。
- 不需要"丢弃重生"逻辑——多 segment 机制天然处理 abort 后续。

**P2 语义修正**（partial_rollout 下原表述不成立）：
- 原表述："group 内 sibling 轨迹尽量不跨权重版本"——在 partial_rollout
  下，单条 trajectory 本身就可能跨多个 segment（多个权重版本），这是
  partial_rollout 的固有特性，无法避免。
- 修正为："**group 内 sibling 的 segment 集合尽量一致**"——即 sibling
  的被打散节奏趋同（都在 V1+V2，而不是 A1 在 V1+V2、A2 在 V2+V3）。
- group affinity 的作用：通过集中 retract/admit 同一 group 的 sibling，
  让 sibling 的被打散节奏一致 → segment 集合趋同 → group 内版本分布
  更均匀。
- 这个修正需要同步更新主设计文档 §1 P2。

---

## Q12: multi-segment resume 下 token_versions 语义

**问题**：P1 用 `token_versions` 度量 staleness，但 resume 后新 segment 的
token_versions 如何赋值？可能产生 staleness 统计偏差。

**待定**：
- 新 segment token_versions 赋值规则

**决议**：**现有 token_versions 赋值已经是正确语义，不需要修改。**

理由：
- **入口**：`_populate_new_rollout_fields`（`llm_server.py:333`）从 segments
  列表构建 token_versions。
- **segment 版本号来源**：每个 segment 的 `global_steps` 由 SGLang server 侧
  在 segment 完成时填入（`async_sglang_server.py:696`
  `extra_fields = {"global_steps": self.global_steps}`），值为**该 segment
  生成时的权重版本**。
- **展开逻辑**：`build_token_versions`（`reprefill_utils.py:173`）把 segment
  版本号展开到该 segment 的每个 token：
  ```python
  for v, n in zip(segment_versions, segment_lengths):
      tokens.extend([int(v)] * n)
  ```
- **multi-segment 语义**：
  - segment 1（abort 前生成）：`global_steps` = 旧版本。
  - segment 2（abort 后用新权重继续）：`global_steps` = 新版本。
  - token_versions = [旧版本]*len(seg1) + [新版本]*len(seg2)。
- **staleness 度量正确性**（`reprefill_utils.py:280`
  `gap = current_version - tv[j]`）：
  - 前段 token gap = current - 旧版本（大，旧权重生成）。
  - 后段 token gap = current - 新版本（小，新权重生成）。
  - 这正是 P1 想要的 token 粒度 staleness 度量，没有统计偏差。
- **P1 度量在 multi-segment 下正确成立**。
- **不需要改 default 0 → assert**：正常路径不会触发 default，保持现状。

---

## Q13: P2 group 内聚性度量指标

**问题**：P0 有 KV 利用率、P1 有 `offpolicy_token/*`，但 P2 "group 内聚性"没有对应 metric。
至少需要一个 `group_staleness`（group 内 sibling 跨多少个权重版本）才能判断 P2 是否达成。

**待定**：
- group_staleness metric 定义
- 采集点

**决议**：

- **度量选择**：`group_version_set_jaccard_mean`（唯一能精确反映 P2 修正后语义的度量）。

- **P2 修正后语义回顾**（Q11）：group 内 sibling 的 segment 集合尽量一致。
  - "一致"有两个层面：
    - 层面 1：跨度一致（sibling 跨的版本范围大小一致）。
    - 层面 2：位置一致（sibling 跨的具体版本集合一致）。
  - P2 真实目标是层面 2。

- **三类候选度量对比**：
  - `group_staleness_span`（max - min of ∪ sibling token_versions）：
    只反映 group 整体跨版本规模，无法区分"一致跨多版本"（P2 达成）
    和"各跨不同版本"（P2 未达成）。
  - `group_segment_count_std`（sibling segment 数 std）：只反映打散次数
    差异，无法识别"打散次数一致但版本不同"。
  - `group_version_set_jaccard_mean`：精确反映层面 2，唯一能区分所有场景。

- **精确定义**：
  - 对每个 group，收集 n 条 sibling 的 segment 版本集合
    $V_1, V_2, \ldots, V_n$（每条 sibling 的 $V_i$ = 该 sibling 的
    `token_versions` 的 unique 集合）。
  - 计算所有 sibling 两两 Jaccard 相似度：
    $J_{ij} = |V_i \cap V_j| / |V_i \cup V_j|$
  - group 一致性 = 所有 $J_{ij}$ 的均值（$i < j$，共 $C(n,2)$ 对）。
  - batch 聚合：batch 内所有 group 的 Jaccard 均值的平均。

- **metric 名**：`offpolicy_token/group_version_set_jaccard_mean`。
  - 值域 [0, 1]，越接近 1 → P2 越达成。

- **边界处理**：
  - group 只有 1 条 sibling（n=1）：无两两配对，返回 1.0（视为一致）。
  - 两条 sibling 都没被打散（version set 都是单元素 {v}）：Jaccard = 1.0。

- **采集点**：扩展 `compute_and_emit_token_staleness_metrics`
  （`reprefill_utils.py:201`）：
  - batch 内每条 trajectory 的 `token_versions` 已存在（Q12 确认）。
  - batch 内每条 trajectory 的 `group_id` 通过 Q6/Q7 的传递链路获得。
  - 按 group_id 分组 + 两两 Jaccard 计算。

- **辅助 metric**：不做 `group_staleness_span` —— 信息量不足，容易误导。

---

## Q14: group 路由亲和（verl LB 层）

**问题**（实施 group_id 传递时发现）：
- verl `GlobalLoadBalancer`（`llm_server.py:81 acquire_server`）用
  `request_id` 做 sticky session key。
- 现状每条 sibling 在 `single_turn_agent_loop.py:66` 用 `uuid4().hex`
  独立生成 `request_id`，因此同 group 的 n 条 sibling **会被分散到不同
  engine replica**。
- Q8/Q9 的 group affinity 是 **SGLang scheduler 层**逻辑，前提是
  sibling 在同一 engine 内被调度。若 verl LB 不做 group 亲和，scheduler
  层的 affinity 调度效果会大打折扣——sibling 根本不在同一 scheduler 视野内。

**决议**：**LB 用 group_id 做 sticky session key**（而非 request_id）。

理由：
- group_id 本来就是"逻辑标识"，天然适合做 sticky key。
- 同 group sibling 集中到同一 engine → scheduler 层 group affinity 才能
  充分发挥。
- 单 engine KV 压力集中的风险由 scheduler 层 retract affinity 自然处理
  （§3.1 集中 retract 同 group 本来就是语义）。
- group_id 跨 multi-segment resume 保持稳定（Q10），所以 sticky 也跨
  resume 生效——同一 request 的所有 segment 路由到同一 engine，符合
  partial_rollout 的 KV 复用需求。

**实施点**：
- `LLMServerClient.generate`（`llm_server.py:230`）在调
  `acquire_server(request_id=...)` 时，改用 `group_id`（若存在）作为
  sticky key；fallback 到 `request_id`（group_id 为 None 时，如 distillation
  teacher 等非 group 路径）。
- `FullyAsyncLLMServerClient.generate` 继承同一行为，multi-segment resume
  循环里每次 `super().generate()` 都传同一 `group_id`，sticky 自然稳定。
- `GlobalLoadBalancer` 本身不需要改——它只看传入的 key 字符串，不关心
  是 request_id 还是 group_id。

---

## Q15: step 边界硬切换 — 攒够 batch 数 group 即停止 rollout，partial group 留到下 step

**group 内 sibling 的独立性**（前提）：
- group 内 `rollout.n` 条 sibling **各自独立**向 SGLang 发 generate
  请求，独立生成、独立被 retract、独立 multi-segment resume。
- sibling 之间在 rollout 阶段无耦合——A sibling 被 retract 不影响
  B sibling 的生成。
- **唯一耦合点**：group 内所有 `rollout.n` 条 sibling 都 `finished`
  后，整个 group 才能被 trainer 消费进入训练。`_classify_cases`
  （`trainer_partial_reprefill.py:284-289`）按 group 聚合判
  `all_stale` / `partial_stale`。

**`all_stale` 的精确语义**（来自 `_classify_cases`）：
- `group_stale[uid]` = 该 group 里 `reason == "stale"` 的 sibling 数
  （`stale` 指该 sibling 被 resume 过，跨了权重版本）。
- `all_stale`: `stale == total`（group 内所有 sibling 都 stale）。
- `partial_stale`: `stale < total`（含 0 个 stale 的 group）。
- `all_stale == 0` 的等价表述：每个被消费的 group 至少有 1 个 sibling
  是本 step 完整生成（没 resume 过）。

**问题**：当前 `partial_reprefill` 的 rollout 持续时间长（跑到所有
group 完成或外部触发条件才进训练）。rollout 时间长 → SGLang KV 紧张
频次高 → sibling 被 retract/resume 机会多 → stale token 累积 →
`all_stale` 比例高。虽然 P2 度量 `group_version_set_jaccard_mean`
高（sibling 一致 stale），但训练 step 里大量"全 stale group"是
训练质量的不利信号。

**目标新语义**：
- **攒够 `train_batch_size` 个完整 group 时，立即触发 rollout→train
  切换**：调 `abort_all` 清空 SGLang 在飞 request（含 partial group
  的 sibling），已完成的 group 进入本 step 训练 batch。
- **partial group 不丢弃**——被 abort 的 partial group sibling 通过
  multi-segment resume 在下一 step 续传（已生成部分作为新 prompt
  重新发给新权重的 SGLang），最终被后续 step 消费。
- **缩短 rollout 持续时间** → 减少 retract/resume 频次 → 减少
  stale token 累积 → `all_stale` 自然下降。Q15 不强制 `all_stale == 0`
  （偶发噪声可接受），但通过缩短 rollout 时间使其趋近 0。
- **允许 `partial_stale`**：group 内部分 sibling 跨 step 边界、
  下一 step 完成，是 OK 的。

**时序图**（Q15 跨 step 流程，以 step N → step N+1 为例）：

```
    verl trainer                  verl rollout client              SGLang engine
    (rollout poll)                (LLMServerClient)                (Scheduler)
        │                                │                              │
   ─────┼──────── step N rollout 阶段 ───┼──────────────────────────────┼─────
        │  发新 group g1..gk             │  generate(g1.s1..sk.sn)      │
        │  (每 group n 条 sibling)       │ ───────────────────────────► │ prefill+decode
        │                                │                              │ g1.s1 finished
        │                                │ ◄──────────────────────────  │ g1.s2 finished
        │  poll replay buffer            │                              │ ... g1.sn finished
        │  g1 完整 ✓                     │                              │
        │  poll... g2 完整 ✓             │                              │ g2.s1 retracted
        │                                │                              │   ↳ waiting_queue
        │                                │                              │ g2.s2 finished
        │  ...                           │                              │ ... g2.sn finished
        │  g2 完整 ✓ (g2.s1 partial)     │                              │
        │  poll...                       │                              │ g3.s1 generating
        │  g3 partial (s1 在飞)          │                              │ g3.s2 generating
        │                                │                              │
        │  完整 group 计数 == batch_size │                              │
        │  触发切换 ────────────────────►│ abort_all() ──────────────► │ 清空 waiting_queue
        │                                │                              │ 清空 running_batch
        │                                │                              │ 清空 KV cache
        │                                │                              │ (g3.s1/s2 被 abort)
        │                                │                              │
   ─────┼──────── step N train 阶段 ─────┼──────────────────────────────┼─────
        │  用 batch_size 个完整 group 训练                              │
        │  (含 g2.s1 partial stale)                                     │
        │  得到新参数 W_{N+1}                                           │
        │                                │                              │
        │  参数同步 ──────────────────────────────────────────────────► │ 权重更新 → W_{N+1}
        │                                │                              │
   ─────┼──────── step N+1 rollout 阶段 ─┼──────────────────────────────┼─────
        │  通知 rollout 恢复            │                              │
        │  ◄─────────────────────────────│                              │
        │                                │  resume g3.s1                │
        │                                │  (已生成 prefix + 新 prompt) │
        │                                │ ───────────────────────────► │ 用 W_{N+1} prefill prefix
        │                                │                              │   ↳ radix cache 命中
        │                                │                              │   ↳ 继续生成剩余 token
        │  发新 group g4..               │  generate(g4.s1..)           │
        │                                │ ───────────────────────────► │ prefill+decode
        │                                │                              │ g3.s1 finished
        │                                │ ◄──────────────────────────  │ g3.s2 finished
        │  poll... g3 完整 ✓             │                              │
        │  (g3.s1/s2 跨 step N+1 完成，   │                              │
        │   partial_stale)               │                              │
        │  ...                           │                              │
```

**时序图关键点**：
- **step N rollout**：发 g1..gk，g1/g2 依次完整，g3 partial（s1/s2 还在飞）
- **触发切换**：完整 group 计数 == `train_batch_size`（图中 g1+g2==2，假设 batch_size=2）→ 调 `abort_all`
- **`abort_all` 行为**：清空 SGLang waiting_queue / running_batch / KV cache；g3.s1/s2 被 abort
- **step N train**：用 g1+g2 训练（g2.s1 是 partial_stale，跨了 resume_version），得到 W_{N+1}
- **参数同步**：W_{N+1} 同步给 SGLang
- **step N+1 rollout**：
  - g3.s1/s2 通过 multi-segment resume 续传（已生成 prefix 作为新 prompt 重新发给 W_{N+1} 的 SGLang，radix cache 命中 prefix）
  - 同时发新 group g4
  - g3 完成 → 写入 replay buffer，被 step N+1 或后续 step 消费（partial_stale）


**与 Q8/Q9 retract/admit affinity 的关系**：
- **Q8 retract affinity 语义不变**：SGLang 内部 retract 时优先
  retract 已有 sibling 被 retract 的 group——目的是"集中 retract
  同 group，让被 retract的 group 整体推迟、让其他 group 优先用 KV
  资源完成"，即**优先保证 group 粒度的完成**。
- **Q9 admit affinity 语义不变**：admit 时优先让已有 sibling 在飞
  的 group 继续 admit，让 group 尽快完成。
- Q15 不改动 SGLang 侧的 affinity 行为。Q15 只在 trainer 侧加一个
  触发条件：攒够 `train_batch_size` 个完整 group 就停止本 step 的
  rollout 消费。

**与 Q11 决议的关系**：
- Q11 决议：`abort_all` 后 multi-segment resume 继续生成，group 跨
  segment 完成。这是现有机制。
- Q15 依赖 Q11——partial group 的 sibling 被 `abort_all` 中断后，
  靠 multi-segment resume 在下一 step 续传。Q15 不改这个机制，只改
  触发 `abort_all` 的条件。
- Q15 不冲突 Q11，也不冲突 Q8/Q9。

**与 `max_resume_staleness` 的关系**：
- Q15 模式下 `max_resume_staleness` 仍有意义——partial group 跨
  step 续传，需要限制最多跨多少 step（避免老 group 永远完不成）。
- 语义不变：限制单条 trajectory 跨多少权重版本。

**与 P2 度量 `group_version_set_jaccard_mean` 的关系**：
- Q15 模式下该度量仍是 P2 判据——`partial_stale` group 的 sibling
  version set 部分重叠（有的 sibling 本 step 完成，有的跨 step 完成），
  Jaccard 介于 0 和 1 之间。
- `all_stale` 下降 + `jaccard_mean` 保持高位 = P2 目标达成（sibling
  跨版本的版本集合尽量一致，且 group 内至少有 fresh sibling）。

**实施方向（待定细节）**：

- **当前 rollout/train 串行流程**（Q15 不改串行结构，只改触发条件）：
  1. **rollout 阶段**：verl 发 generate 请求给 SGLang，SGLang 生成轨迹。
  2. **切换触发**：攒够 `train_batch_size` 个完整 group → 调
     `abort_all` 清空 SGLang 所有在飞 request（含 partial group 的
     在飞 sibling）。
  3. **train 阶段**：用这批完整 group 训练，得到新参数。
  4. **参数同步**：新参数同步给 SGLang（权重更新）。
  5. **下一 step rollout 恢复**：
     - **partial group 的 sibling 续传**：被 abort 的 partial group
       sibling 通过 multi-segment resume 续传——已生成部分作为新
       prompt 重新发给新权重的 SGLang，继续生成剩余 token
       （`FullyAsyncLLMServerClient.generate` 的 multi-segment while
       循环天然处理）。
     - **本 step 也可以发新 group**（取决于配置和 KV 容量）。

- **`abort_all` 必须调**：不是可选——它清空 SGLang 状态以接收新权重。
  partial group 的 sibling 被 abort 后不丢弃，靠 multi-segment resume
  在下一 step 续传。

- **触发点**：trainer 侧 replay buffer poll loop
  （`trainer_partial_reprefill.py` 的 sample 消费路径）。
  当前流程已经在某个条件触发 `abort_all` + 切换 train；Q15 改的是
  触发条件——从"等所有 group 完成（或当前的其他条件）"改为"攒够
  `train_batch_size` 个完整 group"。

- **"完整 group" 判定**：group 的所有 `rollout.n` sibling 都已
  `finished`。不要求 `resume_version == current_step_version`——
  允许 partial stale（部分 sibling 跨 step 完成）。

- **partial group 跨 step 续传**：在飞 sibling 的
  `FullyAsyncLLMServerClient.generate` 多 segment 循环在 `abort_all`
  后被中断，下一 step 用新权重重新发请求（已生成部分作为新 prompt），
  继续生成剩余 token。完成后写入 replay buffer，等待后续 step 消费。
  这是 Q11 决议的现有机制，Q15 不改。

- **SGLang 侧无改动**：retract_decode / get_new_batch_prefill 的
  affinity 行为保持 Q8/Q9 原方向（集中 retract/admit 同 group，
  优先保证 group 粒度完成）。

**代价**：
- **partial group 的 prefix 重算开销**：`abort_all` 释放 KV cache，
  下一 step resume 时需要重新 prefill 已生成部分。但 multi-segment
  resume 有 prefix 复用（radix cache 命中），开销不等于全量重算。
  Q11 决议已接受这个开销。
- **step 延迟**：无额外延迟（攒够就停，比"等所有 group 完成"更快
  进入训练）。
- **rollout 吞吐下降**：每 step 只产出 `train_batch_size` 个完整
  group + 若干 partial group，比"跑到所有 group 完成"产出少。但
  这正是 Q15 的目的——用更少的 rollout 时间换取更少的 stale token。
- **跨 step KV 状态**：`abort_all` 清空 KV，下一 step 从冷启动开始
  （partial group resume 靠 radix cache 复用 prefix，不是跨 step KV
  保持）。Q11 已处理。

**与 Q14 LB 路由亲和的协同**：
- Q14 把同 group sibling 路由到同一 engine——Q8/Q9 affinity 依赖
  这一点才能生效。Q15 不直接依赖 Q14（Q15 是 trainer 侧触发逻辑），
  但 Q15 的效果（`all_stale` 下降）依赖 Q8/Q9 affinity 的效果
  （group 粒度优先完成），间接依赖 Q14。

**不在本决议范围**：
- partial group 跨多少 step 过期丢弃——由 `max_resume_staleness`
  控制，本决议不改。
- `train_batch_size` 动态调整——当前固定。
- 跨 step 的 group 优先级调度——下 step 是否优先完成 partial group
  vs 发新 group，由 SGLang admit affinity（Q9）自然处理（partial
  group 的 sibling 已在飞，admit 优先）。
- SGLang 侧 affinity 改动——Q15 不改 SGLang，Q8/Q9 保持原方向。

**待定问题**：
- 触发阈值：严格 `== train_batch_size` 还是 `>=`？严格等于更干净
  （训练 batch 恰好是 batch_size 个 group），倾向严格等于。
- warmup 阶段：`num_warmup_batches` 阶段是否也用 Q15 语义？倾向是，
  保持一致。
- `rollout.n=1` 退化为非 GRPO 场景：Q15 仍适用（group 即单条轨迹），
  `all_stale` 含义退化为"该轨迹是否跨 step"，仍要求趋近 0。

