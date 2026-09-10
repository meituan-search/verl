# 02 — SGLang logprob 机制与 verl piggyback 的衔接

> 核心文件：
> - `third_party/sglang/python/sglang/srt/managers/io_struct.py`（请求参数定义与默认值）
> - `third_party/sglang/python/sglang/srt/managers/schedule_batch.py`（logprob_start_len 绝对→相对换算）
> - `third_party/sglang/python/sglang/srt/layers/logits_processor.py`（窗口行筛选、lm_head、log_softmax）
> - `third_party/sglang/python/sglang/srt/layers/sampler.py`（decode 步 logprob）
> - `third_party/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py`（`[None]+[:-1]` 移位、逐 step 收集）
> - `third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py`（convert_logprob_style 汇总装配 meta_info）
> - `verl/workers/rollout/sglang_rollout/async_sglang_server.py`（verl→SGLang 参数翻译 + 输出消费）
> - `verl/workers/rollout/llm_server.py`（piggyback 字段构建）
>
> 核心问题：
> 1. `prompt_logprobs=0`、`logprob_start_len`、`return_logprob` 三个旋钮分别控制什么？
> 2. 一次 generate 能否同时拿到"最后一个 prompt token 的 logprob"和"全部输出 token 的 logprob"？
> 3. input_token_logprobs 的 entry 语义（含 `[None]+[:-1]` 移位）与不变式是什么？
> 4. logprob 计算的额外开销在哪里，为什么 piggyback 能"近免费搭车"？

---

## 一、传入参数：三个旋钮

verl 适配层（`async_sglang_server.py:596-626`）把 vLLM 风格参数翻译成 SGLang 原生 API：

```python
return_logprob = sampling_params.pop("logprobs", False)
prompt_logprobs = sampling_params.pop("prompt_logprobs", None)
prompt_logprob_start_len = sampling_params.pop("logprob_start_len", None)
if prompt_logprobs is not None:
    return_logprob = True

request = {
    "input_ids": prompt_ids,
    "return_logprob": return_logprob,          # 总开关
    "logprob_start_len": start_len or 0,       # input 侧窗口起点（绝对位置）
    # prompt_logprobs > 0 时才设：
    "top_logprobs_num": prompt_logprobs,       # topk，默认 0
}
```

| 参数 | 选项 | 效果 |
|---|---|---|
| `return_logprob` | False | 什么都不算（最便宜） |
| | True, `logprob_start_len=-1`（默认，io_struct.py:380） | 只算 output 侧（聊天场景） |
| | True, `logprob_start_len=S` | input 侧窗口 `[S, len)` + output 侧 |
| `top_logprobs_num` | 0 | 每位置只返回实际 token 的 logprob |
| | K>0 | 每位置 K+1 个（贵：全词表 topk 选择+序列化） |

**关键点**：`prompt_logprobs=0` ≠ "不算 prompt logprob"，而是"算窗口内每个位置的 logprob，但每位置只返回实际 token 那一个值"。`prompt_logprobs` 的数值只有 topk 用途；窗口大小由 `logprob_start_len` 唯一决定（不传 = -1 = 不算 input 侧）。

## 二、窗口换算：绝对 → 相对（schedule_batch.py:1321-1337）

```python
def set_extend_input_len(self, extend_input_len: int):
    self.extend_input_len = extend_input_len
    if self.logprob_start_len == -1:
        logprob_start_len = len(self.fill_ids)        # 默认贴到序列末尾 → 空窗口
    else:
        # 窗口起点不能落在已被缓存/已算过的前缀里
        logprob_start_len = max(self.logprob_start_len, len(self.prefix_indices))
    self.extend_logprob_start_len = min(              # 窗口起点不能越过本批末端
        logprob_start_len - len(self.prefix_indices),
        self.extend_input_len,
    )
```

坐标系：

```
全序列（fill_ids）:  [0 ..................... prefix ................. prefix+extend_input_len)
                     └───── prefix_indices ─────┘└──── 本批 extend tokens ────┘
logprob_start_len:      绝对坐标（全序列中的位置）
extend_logprob_start_len: 相对坐标（本 extend 批内偏移）
```

- **max 防"窗口起点掉进前缀"**：prefix_indices 是 radix cache 命中或 chunked prefill 已算过的部分，那些位置本轮不产生 hidden state；不 max 会得到负的相对偏移。
- **min 防"窗口起点越过本批末端"**：空窗口时由 logits_processor.py:497-500 接住（`extend_len == start_len` 时退化为只取 1 行算采样 logits，0 个 input logprob）。
- **配套防泄漏**（schedule_batch.py:1015-1016）：请求 logprob 窗口时 `max_prefix_len = min(max_prefix_len, logprob_start_len)` —— 限制 radix cache 最多匹配到窗口起点，防止缓存把想要 logprob 的行吞掉。piggyback resume 场景正靠这条保命。

chunked prefill 数字验证（全序列 8，start_len=2，chunk=4）：

| Chunk | prefix | extend_len | max(2,prefix) | 相对起点 | 产出 input logprobs |
|---|---|---|---|---|---|
| 1 | 0 | 4 | 2 | 2 | 位置 2,3（增量收集） |
| 2 | 4 | 4 | 4 | 0 | 位置 4-7 |

合计 6 = 8-2，覆盖 `[2,8)` 无遗漏无重复。

## 三、计算侧：窗口行筛选（logits_processor.py:489-529）

`_get_pruned_states` 决定哪些 hidden state 过 lm_head：

```python
pruned_states_list.append(hidden_states[pt + start_len : pt + extend_len])
input_logprob_indices.extend([...])  # extend_len - extend_logprob_start_len 行
```

- 不开 return_logprob：每序列只 1 行（采样行），走 logits_processor.py:341-356 快速路径。
- 开 return_logprob：每序列 `(extend_len - start_len) + 1` 行，每额外行 = 一次 lm_head GEMM + 一次全词表 log_softmax。

prefill 前向的 raw 结果（例：`ids=[a,b,c,d]`, start_len=2）：

```
raw = [lp(d|abc), lp(e|abcd)]    # 2 行；lp(e) 是采样 token 的 logprob
```

decode 步（sampler.py:114-117 / scheduler_output_processor_mixin.py:454-455, 539）每步只 append 一个 output logprob，近似免费（softmax 本来就要算）。

## 四、后处理：`[None]+[:-1]` 移位（scheduler_output_processor_mixin.py:677-690）

```python
req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]
req.input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len:]
```

- 采样 token 的 logprob（raw 最后一行）从 input 侧砍掉，转存 `output_token_logprobs_val` 首条（add_logprob_return_values, :901-903）—— 一份计算、两处消费。
- `convert_logprob_style`（tokenizer_manager.py:1979+）把 input/output 两侧同时装配进 meta_info。

**三条不变式**：

1. `input_token_logprobs` 与 `ids[start_len:]` 逐位对齐，entry 0 恒为 None；
2. `len(output_token_logprobs) == len(output_ids) == N`；
3. raw 计算 `len - start_len` 行，最后一行即 output 侧首条，无重复计算。

### 语义澄清：input 侧与 output 侧没有两套定义

两个列表的 logprob 语义**完全统一**：entry 都是被描述位置上 **token 自己的 logprob，由其之前的所有 token 预测**：

```text
input_token_logprobs[i]  = log P( ids[start_len+i] | ids[:start_len+i] )
output_token_logprobs[i] = log P( output_ids[i]    | prompt + output_ids[:i] )
```

`output_token_logprobs[i]` **不是**"当前 token 对下一个位置的预测"——若如此，最后一位将描述一个尚未采样、logprob 无从谈起的 token。代码直接可见：`req.output_token_logprobs_idx.append(next_token_ids[i])`（:903），val 和 idx 是同一个 token，逐位对齐。

容易混淆的根源：一条 logprob 行在位置 p 的 hidden state 上计算，算出的是 `lp(位置 p+1 的 token | tokens ≤ p)` —— 它描述的**永远是 p+1 那个 token 自己**；它进哪个列表，只取决于 p+1 在请求的**哪一侧**（输入/输出），而非另一套定义：

```text
位置:      0    1    2    3    4   ‖   5    6
token:     t0   t1   t2   d0   d1  ‖   x0   x1
                              └──hidden(4) 算出 lp(x0)──┘
──────────────────────────── input 侧 ─┬─ output 侧 ─

lp(d0) = P(d0|t0..t2)   → 描述位置3的 token → 位置3 ∈ 输入 → input_token_logprobs
lp(d1) = P(d1|..d0)     → 描述位置4的 token → 位置4 ∈ 输入 → input_token_logprobs
lp(x0) = P(x0|..d1)     → 描述位置5的 token → 位置5 ∈ 输出 → output_token_logprobs[0]
lp(x1) = P(x1|..x0)     → 描述位置6的 token → 位置6 ∈ 输出 → output_token_logprobs[1]
```

hidden(4) 本身在输入序列里，但它算出的 logprob 描述位置 5 的 token（x0，输出侧）——所以 prefill 时这行 raw 从 input 列表被 `[:-1]` 砍掉、转存 output 列表首条。**"一份计算、两处消费"是按位置归属分拣，不是两套语义。** 唯一的错位感来自 input 侧 entry 0 的 None 占位（窗口起点之前的 token 未计算），那是窗口标记，不是另一种语义。

| 列表 | entry i 的含义 | 对齐锚点 | 窗口起点处 |
|---|---|---|---|
| `input_token_logprobs[i]` | 输入第 i 个窗口 token 自己的 logprob（由其前缀预测） | `ids[start_len:]` | entry 0 = None（未计算） |
| `output_token_logprobs[i]` | 第 i 个输出 token 自己的 logprob（由 prompt+此前输出预测） | `output_ids` | 无 None，首条即第一个输出 token |

## 五、完整例子：piggyback resume（verl 实际用法）

resume 请求：输入 = `[t0,t1,t2] + D[d0,d1]`（纯 prompt 3 + decoded prefix 2，共 5），`logprob_start_len = len(纯P)-1 = 2`，`max_new_tokens=2`，`prompt_logprobs=0`（topk 关）。

窗口 = `[2, 5)`，3 行：`lp(d0|t0t1t2)`, `lp(d1|…d0)`, `lp(x0|…d1)`。

```python
meta_info = {
  "output_ids":              [x0, x1],                            # N=2
  "input_token_logprobs":    [None, (lp(d0), d0), (lp(d1), d1)],  # 对齐 ids[2:]，长度 3
  "output_token_logprobs":   [lp(x0), lp(x1)],                    # 长度 2
}
```

ASCII 全景：

```
位置:      0     1     2     3     4   |  5     6
token:     t0    t1    t2    d0    d1  |  x0    x1
                                   (新输出)
窗口 [2,5):      └──────── input 侧 ────┘└─ output 侧 ─┘
raw 行:                 lp(d0) lp(d1) lp(x0)        lp(x1)  lp(x2)...
input 侧:        None  lp(d0) lp(d1)   ↘ 转存
output 侧:                            lp(x0)        lp(x1)  ...
```

verl 消费链：

```python
# async_sglang_server.py:690-705 — start_len>0 短 list 透传
extra_fields["prefix_prompt_logprobs"] = [(None, t2), (lp(d0), d0), (lp(d1), d1)]

# llm_server.py:348-352 — _build_piggyback_fields
prefix = prefix_prompt_logprobs[1:prefix_len+1]   # [lp(d0), lp(d1)]
suffix = last_seg.log_probs                       # [lp(x0), lp(x1)]（同一次 generate）
new_rollout_log_probs = prefix + suffix           # 全部在 W_resume 权重下
```

**一次 generate 同时拿到 decoded prefix + 新输出的全部 logprob** —— 回答核心问题 2：能，只要 `logprob_start_len ≤ 纯prompt_len - 1`。

## 六、边界情况速查

| 参数组合 | 窗口 | input_token_logprobs | 典型用途 |
|---|---|---|---|
| `start_len = -1`（不传） | 空 | 不返回 | 纯聊天，只要 output logprob |
| `start_len = 0` | 全 prompt `[0,L)` | 长度 L，entry 0 = None | distillation teacher |
| `start_len = L-1` | 仅最后 1 位 | `[None]`，无有效值 | 陷阱：拿不到任何 prompt token 的 logprob |
| `start_len = len(纯P)-1` | `[len(P)-1, L)` | 长度 L-S，entry 0 = None | piggyback（首 entry None 是已知错位，切 `[1:]` 绕过） |
| `max_new_tokens=0` + `start_len=0` | 全 prompt | 全 prompt logprob | trainer case 2 全量 reprefill（output_ids 为空） |

注意 `start_len = L-1`（总长-1）与 `start_len = len(纯P)-1` 的区别：前者窗口只剩采样位，t3 自己的 logprob 根本没被计算；后者才包含最后一个 prompt token。

## 七、开销账

| 阶段 | 开销 | 控制手段 |
|---|---|---|
| prefill 额外行 | lm_head GEMM + 全词表 log_softmax，∝ 窗口长度，**主导项**（Qwen3-8B 每行 ≈1.25 GFLOP） | `logprob_start_len` 收窄窗口；`enable_logprobs_chunk` 分块控显存（logits_processor.py:367-371, 665-834） |
| prefill topk | 全词表 topk 选择+序列化 | verl 用 `prompt_logprobs=0` 避开 |
| decode | ~0（每 token 1 行 log_softmax，softmax 本来就算） | 无需 |
| CPU/序列化 | logprob list 逐 token 传输 | 量小 |

piggyback 近免费的原理：resume 时 KV 已失效、重新 prefill 是沉没成本；把 start_len 从 0 抬到 `len(纯P)-1` 后，额外行从"整个 prompt"缩到"仅 decoded prefix + 1"——这就是 llm_server.py:509-511 注释所说 "computing them is the dominant cost on long-prompt datasets (dapo)" 的应对。

## 设计要点小结

| 要点 | 位置 | 说明 |
|---|---|---|
| `prompt_logprobs=0` 的语义 | async_sglang_server.py:601-626 | 非 None → return_logprob=True；数值仅作 topk；窗口由 logprob_start_len 决定 |
| 默认 -1 = 空窗口 | io_struct.py:379-380, schedule_batch.py:1329-1330 | 不传 logprob_start_len 则 input 侧零开销 |
| max 钳制 | schedule_batch.py:1333 | 窗口起点不能掉进缓存/已算过的前缀（防负偏移） |
| min 钳制 | schedule_batch.py:1334-1337 | 窗口起点不能越过本批末端（空窗口退化为纯采样） |
| radix cache 防泄漏 | schedule_batch.py:1015-1016 | max_prefix_len 被压到 logprob_start_len 以内 |
| `[None]+[:-1]` 移位 | scheduler_output_processor_mixin.py:678 | entry 0 恒 None；采样 logprob 转存 output 侧首条 |
| 一份计算两处消费 | :901-903 + :678 | raw 最后一行同时是 output_token_logprobs[0] |
| 长度不变式 | async_sglang_server.py:651-663 | len(output_token_logprobs)==len(output_ids)，verl 靠它校验、不符则两侧清空 |
| piggyback 窗口选择 | llm_server.py:515-520 | start_len = len(纯prompt)-1，prefix 切 `[1, prefix_len+1)` |

**一句话**：SGLang 的 logprob 开销本质是"把原本只需采样 1 行的 lm_head 变成多算窗口内 N 行全词表 log_softmax"——窗口由 `logprob_start_len` 唯一决定，它既是功能旋钮也是性能旋钮；verl 的 piggyback 把窗口缩到 decoded prefix，让 resume prefill 的 logprob 变成近免费的搭车。

---

# 附录 A：FullyAsyncLLMServerClient.generate 完整生命周期（piggyback 视角）

> 核心文件：
> - `verl/workers/rollout/llm_server.py`（:360-534 循环主体；:281-357 `_build_piggyback_fields`）
> - `verl/checkpoint_engine/base.py`（:486-538 权重同步 8 步，pause 的来源）
> - `verl/workers/rollout/sglang_rollout/async_sglang_server.py`（参数翻译 + 输出消费）
> - `verl/trainer/ppo/v1/reprefill_utils.py`（token_versions / 拼接 / decide_case）
>
> 核心问题：一条轨迹经历多次权重同步打断后，`final_output`（含 `new_rollout_log_probs`）是如何一步步产生的？

## A.1 角色与调用链

```
AgentLoopWorkerTQ.generate_sequences (agent_loop_tq.py:60)        ← trainer 下发 batch
   └─ AgentLoopWorker._run_agent_loop (agent_loop.py:669-702)
        ├─ hydra.utils.instantiate(_agent_loop_registry["single_turn_agent"])
        │     # agent_name 由 default_agent_loop 解析（rollout.yaml:245 默认值）
        └─ SingleTurnAgentLoop.run (single_turn_agent_loop.py:38-115)
             ├─ apply_chat_template / ct_build_initial_tokens     # 模板化 prompt
             ├─ output = await self.server_manager.generate(      # ← FullyAsyncLLMServerClient
             │       request_id, prompt_ids, sampling_params, ...)
             │       # piggyback/partial_rollout 的一切都发生在这个 await 内部
             └─ AgentLoopOutput:
                  prompt_ids / response_ids[:response_length]     # 截断到预算
                  response_logprobs = output.log_probs            # 混合版本 decode logprob
                  extra_fields = output.extra_fields              # ★ 原样透传三件套
                  ▼
        _agent_loop_postprocess (agent_loop_tq.py:151-)           # TQ put（tag + jagged fields）

FullyAsyncLLMServerClient.generate (llm_server.py:401-)           ← 被 SingleTurnAgentLoop 调用
   └─ while True: super().generate()            (LLMServerClient, llm_server.py:229-278)
        ├─ _acquire_server(request_id)           ← GlobalRequestLoadBalancer 粘性路由
        ├─ server.generate.remote(uuid4().hex)   ← SGLangHttpServer actor（每段换新 rid）
        └─ finally: _release_server(server_id)   ← fire-and-forget 计数递减

触发 pause 的一方（与上完全并行）:
FullyAsyncTrainer._fit_update_weights (:690)
   └─ checkpoint_engine.update_weights (base.py:486-538)
```

两个要点：

- **SingleTurnAgentLoop 对 piggyback 完全无感知**：只认 `output.token_ids / log_probs / extra_fields` 三个出口，resume 循环、段合并、蒸馏全部封装在 client 的 generate await 内部——这正是 FullyAsyncLLMServerClient docstring（:350-352）说的 "making rollout interruption invisible to the AgentLoop"。
- **`extra_fields` 原样透传**（single_turn_agent_loop.py:109）：client 蒸馏出的 token_versions / piggyback_marker / new_rollout_log_probs / resume_version 不做加工直接进 AgentLoopOutput.extra_fields。整条链路上唯一理解 piggyback 语义的只有两处——client（生产）和 trainer（消费）。

## A.2 入口准备（llm_server.py:428-448）

```python
prompt_ids = normalize_token_ids(prompt_ids)          # P，全程不变
enable_piggyback = sampling_params.pop("enable_piggyback", False)  # 弹出，不下传后端
original_max_tokens = sampling_params[limit_key]      # 跨段总预算
final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)
segments = []                                          # 每段原始 TokenOutput，收尾蒸馏用
```

| 状态 | 分工 |
|---|---|
| `final_output` | 面向 AgentLoop 的累积结果 |
| `segments` | 面向 piggyback 的原始素材（收尾才蒸馏） |
| `original_max_tokens` | 保证重试段不重复计额度（每轮扣减） |
| `enable_piggyback` | 只在本类内生效，SGLang 不认识这个 key |

## A.3 逐段时间线（piggyback 版，P=[t0,t1,t2]，预算 12，两次打断）

三列分别对应：**client 循环里发生什么 / 此时 SGLang 权重与请求形态 / 循环收到的返回**。

```
┌──────┬────────────┬──────────────────────────────────┬─────────────────────────────────────┐
│ 阶段  │ 权重状态    │ client 循环                       │ SGLang 侧                           │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ seg1 │ W1         │ super().generate(P)              │ prefill(P)@W1                       │
│      │            │ 请求参数(翻译后):                  │ decode 产出 d0..d4                   │
│      │            │  return_logprob=True             │  output_token_logprobs=[lp@W1 × 5] │
│      │            │  logprob_start_len=-1 (未传)     │  input_token_logprobs 不返回        │
│      │            │  top_logprobs_num=0 (未设)       │   (窗口 [4,4) 为空)                 │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ pause│ abort→W2   │ ——（await 中，被动等待）——          │ pause_generation(abort) 打断在飞请求  │
│  #1  │ NCCL 同步   │                                  │ kv释放→NCCL(W2)→kv恢复→resume(8步)   │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ seg1 │ 收到返回    │ stop_reason="aborted" 且可重试     │ 返回已生成部分:                       │
│ 收尾  │            │ 合并: token_ids += [d0..d4]      │  token_ids=[d0..d4]                 │
│      │            │       log_probs += [lp@W1 × 5]   │  log_probs=[lp@W1 × 5]              │
│      │            │ 注入(叠加到现有参数):               │  (无 prompt_logprobs 字段)           │
│      │            │  prompt_logprobs=0               │                                     │
│      │            │  logprob_start_len=len(P)-1=2    │                                     │
│      │            │ sleep(1) 等权重同步落地            │                                     │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ seg2 │ W2         │ super().generate(P+[d0..d4])     │ re-prefill(8 token)@W2 全量前向      │
│      │            │ 请求参数(翻译后):                  │ ★KV 因权重变更已失效，必须重算          │
│      │            │  return_logprob=True             │ ★顺带算窗口[2,8)的 input logprob      │
│      │            │  logprob_start_len=2             │ decode 产出 d5..d9                   │
│      │            │  top_logprobs_num=0              │                                     │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ pause│ abort→W3   │ ——同 pause #1——                  │ 同上，W3                             │
│  #2  │            │                                  │                                     │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ seg2 │ 收到返回    │ 合并: token_ids += [d5..d9]       │  token_ids=[d5..d9]                 │
│ 收尾  │            │       log_probs += [lp@W2 × 5]   │  log_probs=[lp@W2 × 5]              │
│      │            │       prompt_logprobs ← 覆盖      │  prefix_prompt_logprobs=            │
│      │            │       (seg2 的,即 lp(d0..d4)@W2)  │   [None, lp(d0)@W2, .. lp(d4)@W2]   │
│      │            │ 再次注入(同参数,与已生成长度无关)     │                                     │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ seg3 │ W3         │ super().generate(P+[d0..d9])     │ re-prefill(13 token)@W3 全量前向     │
│      │            │ 请求参数(翻译后):                  │   窗口[2,13): input_token_logprobs   │
│      │            │  return_logprob=True             │   = [None, lp(d0)@W3, .. lp(d9)@W3] │
│      │            │  logprob_start_len=2             │ decode 产出 x0, x1                   │
│      │            │  top_logprobs_num=0              │   output_token_logprobs=[lp(x0),    │
│      │            │                                  │    lp(x1)]@W3                       │
├──────┼────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ seg3 │ 收到返回    │ len=12 ≥ 预算 → stop="length"     │  token_ids=[x0,x1]                  │
│ 收尾  │            │ break 退出循环                    │  prefix_prompt_logprobs=            │
│      │            │                                  │   [None, lp(d0..d9)@W3] (11 项)     │
├──────┴────────────┴──────────────────────────────────┴─────────────────────────────────────┘
│ 收尾蒸馏: _build_piggyback_fields(segments, prompt_len=3)
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

注：
- pause 窗口 = checkpoint_engine.update_weights 的 8 步（base.py:486-536）：abort → 临时 worker group → release kv → build PG → NCCL 同步 → finalize → resume kv → resume_generation。客户端在整个窗口内 await 挂起。
- 注入动作发生在**每次确认 abort 之后、重试之前**（:515-521），即重试段的后端权重必然已是新版本 W_resume。
- `logprob_start_len` 恒为 `len(P)-1`（不随已生成长度变化）——窗口起点锚定在纯 prompt 末尾。
- **首段参数的来源**：入口 sampling_params 由 AgentLoop 组装（agent_loop.py:590-604），含 `logprobs=config.calculate_log_probs`（rollout.yaml:233 默认 True）→ 适配层翻译为 `return_logprob=True`；不含 `prompt_logprobs`/`logprob_start_len` → SGLang 默认 `-1`（io_struct.py:379-380）→ input 窗口为空（`start_len = len(fill_ids)`，schedule_batch.py:1329-1330）。这是 rollout_log_probs 的来源，staleness 分析的基准。首段不请求 prompt 侧 logprob，因为首段没有 decoded prefix，算纯 P 的 logprob 纯属浪费（llm_server.py:509-511："the prompt is given, not a sampling decision"）。
- **注入是增量叠加而非替换**：`{**sampling_params, "prompt_logprobs": 0, "logprob_start_len": ...}`（:516-520）；`prompt_logprobs=0` 翻译后 `top_logprobs_num` 保持默认 0（仅当 `>0` 才设，async_sglang_server.py:625-626），decode 侧的 `logprobs=True` 全程保留。

## A.4 每段合并语义（:463-487）——三个"覆盖"

```python
final_output.token_ids.extend(output.token_ids)     # 累积
final_output.log_probs.extend(output.log_probs)     # 累积：各段 decode logprob（各自的 W）
if "prompt_logprobs" in output.extra_fields:
    final_output.extra_fields["prompt_logprobs"] = ...  # ★覆盖★ 只留最后一段
sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)  # 扣预算
```

**prompt_logprobs 必须覆盖的原因**：seg2 的 `lp(d0..d4)@W2` 在 W3 下已过期、无意义。piggyback 只需要最后一段的前缀 logprob，覆盖语义天然淘汰旧段。routed_experts 也是类似处理（首段保留 + 后续段只 concat 新 token 部分，resume 段权重不同路由不可比）。

## A.5 final_output 的收尾蒸馏（:523-529 → `_build_piggyback_fields` :281-357）

三步：

**① token_versions（无条件产出，:302-304）**：

```python
segment_versions = [1, 2, 3]          # 各段 decode 时的 global_steps
segment_lengths  = [5, 5, 2]
token_versions   = [1,1,1,1,1, 2,2,2,2,2, 3,3]   # 逐 token 版本指纹
```

**② piggyback 两种模式（:308-317）**：

- **不需要**（`enable_piggyback=False`，或 `len(segments) < 2` 即从未 resume）：只产出 token_versions，**不获取、不生成** `new_rollout_log_probs`，`piggyback_marker=False`——resume 分支本来就没请求 prompt logprob，无字段是正常的，不报错。
- **需要**（`enable_piggyback=True` 且 ≥2 段）：最后一段必然是携带 prompt_logprobs 请求的 post-abort 重试段，其 `prefix_prompt_logprobs` 的存在与形状是**硬契约**——缺字段（`assert last_pl is not None`）或长度不足（`len(last_pl) < prefix_len+1` → raise）直接报错，绝不静默降级、绝不产出错误形状的数据。拼接后还校验 `len(new_rollout) == prefix_len + len(last_seg.token_ids)`（= 总生成 token 数）。

**③ 拼接 new_rollout_log_probs（:328-353）**：

```python
prefix_len = sum(len(s.token_ids) for s in segments[:-1])     # = 10（seg1+seg2 的产出）
# last_pl = seg3 的 prefix_prompt_logprobs = [None, lp(d0)@W3, ..., lp(d9)@W3]，长度 11
prefix = [last_pl[i][0] for i in range(1, prefix_len + 1)]    # 切 [1,11) → 跳过 None
suffix = [float(x) for x in last_seg.log_probs]               # [lp(x0)@W3, lp(x1)@W3]
new_rollout_log_probs = prefix + suffix                       # 12 项，全部 @W3
resume_version = segment_versions[-1]                         # = 3
```

## A.6 最终 final_output 全貌与三种 logprob 对比

```python
final_output = TokenOutput(
    token_ids   = [d0..d4, d5..d9, x0, x1],          # 12 个
    log_probs   = [lp@W1×5, lp@W2×5, lp@W3×2],       # 混合版本（各段 decode 时点）
    stop_reason = "length",
    extra_fields = {
        "token_versions":        [1×5, 2×5, 3×2],              # 逐 token 版本
        "piggyback_marker":      True,
        "new_rollout_log_probs": [lp(d0..d9)@W3, lp(x0)@W3, lp(x1)@W3],  # 单一权重 W3
        "resume_version":        3,                              # == token_versions[-1]
        "global_steps": 3, "min_global_steps": 1, "max_global_steps": 3,
    })
```

| 集合 | 权重 | 用途 |
|---|---|---|
| `log_probs`（→ rollout_log_probs） | 混合 W1/W2/W3 | 各 token 采样时点的真实分布（staleness 分析基准） |
| `new_rollout_log_probs` | 统一 W3（=resume_version） | π_new-rollout —— trainer 以之替代 old_log_probs 算 ratio，消除版本混杂 |
| case 3（copy） | —— | `token_versions[-1] == 当前参数版本`（整条轨迹都在最新权重下）→ 直接复制 rollout_log_probs；本例不满足 |

下游分派：`decide_case`（reprefill_utils.py:138-156）按 `piggyback_marker`/`token_versions[-1]`/当前参数版本选 case 1/2/3；case 2 由 trainer 用 `reprefill_trajectories`（`max_new_tokens=0, prompt_logprobs=0`, :74-86）全量 re-prefill 重算。

## A.7 设计成立的三个前提

1. **abort 后 KV 必然失效**（权重变了）→ resume 段 re-prefill 是躲不掉的沉没成本；
2. **re-prefill 的 hidden state 正是算 logprob 需要的** → `logprob_start_len=len(P)-1` 只是把"已在算的行"多接一行 lm_head + log_softmax，窗口恰好只覆盖 decoded prefix D（P 的部分被 `[None]` 占位后切掉）；
3. **最后一段的前缀+后缀都在 W_final 下** → 两部分 logprob 权重一致，可直接拼接成完整 `new_rollout_log_probs`；中间段的前缀 logprob 虽也顺带算过（seg2 也注入了参数），但因权重过期被覆盖语义丢弃。

## A.8 跨段状态累积总表

| 字段 | 跨段策略 | 理由 |
|---|---|---|
| token_ids / log_probs | extend 累积 | 每段都是有效生成，各段 decode logprob 供 staleness 分析 |
| routed_experts | 首段保留 + 后续段只 concat 新 token | resume 段权重不同，路由不可比 |
| num_preempted | 累加 | 统计量 |
| stop_reason | 每段覆盖 | 只有最后一段的 stop 才是终局 |
| prompt_logprobs | **覆盖**（只留最后一段） | 只有最后一段的前缀 logprob 在 W_final 下有效 |
| min/max_global_steps | 逐段 min/max | 刻画该轨迹的 off-policy 程度 |
| global_steps | 取最后一段 | resume_version 的来源 |

**一句话**：generate 循环把"多次 pause/resume"折叠成 segments 序列，`_build_piggyback_fields` 再把 segments 蒸馏成三件套——`token_versions`（逐 token 权重版本指纹）、`new_rollout_log_probs`（借最后一次 re-prefill 免费搭车得到的 W_final 统一 logprob）、`resume_version`（该 logprob 的权重版本号）——让 trainer 同时拿到"混杂版本的采样分布"与"单版本的重估分布"，供 ratio 计算与 staleness 分析各自取用，全程对 AgentLoop 透明。
