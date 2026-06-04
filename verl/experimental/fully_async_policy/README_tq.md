# Fully Async Policy with TransferQueue (TQ)

## 概述

本方案在 `fully_async_policy` 基础上，将数据传输通道从 Ray **MessageQueue** 迁移到 **TransferQueue (TQ)**，
同时训练侧通过**多继承**复用 `main_ppo_sync.py` 的 `PPOTrainer` 以直接使用 TQ 原生的 `KVBatchMeta` 训练流程。

### 核心设计原则

1. **最小改动**: `FullyAsyncAgentLoopManager` 整体不动，保持现有推理生成逻辑；仅通过 `FullyAsyncAgentLoopManagerTQ` 轻量适配
2. **ReplayBuffer 控制流速**: 用 ReplayBuffer (Ray Actor) 的 Dual-Layer Slot 机制替代原 `_should_pause_generation`
3. **TQ 替换 MessageQueue**: 数据走 TQ 零拷贝通道，元数据走 ReplayBuffer
4. **Trainer 多继承 PPOTrainer**: 通过 `class FullyAsyncTrainerTQ(PPOTrainer, FullyAsyncTrainer)` 复用 TQ 训练流程

### 核心目标

1. **零拷贝传输**: 使用 TQ 替代 MessageQueue，避免 `ray.cloudpickle` 序列化开销
2. **源头限流**: 在 dataloader 数据获取处 (`_feed_samples`) 通过 `acquire_slot()` 控制生产速度
3. **背压控制**: Dual-Layer Slot 机制限制 in-flight 请求数，无需额外的暂停/恢复逻辑
4. **复用成熟代码**: Trainer 侧多继承 PPOTrainer，直接使用 TQ 原生的 batch 训练流程（_compute_old_log_prob, _
   compute_advantage 等）

## 架构对比

### 现有架构 (MessageQueue + SeparateRayPPOTrainer)

```mermaid
graph TB
    subgraph Rollouter[Rollouter - SeparateRayPPOTrainer]
        RM[FullyAsyncRollouter]
        ALM[FullyAsyncAgentLoopManager]
    end

    subgraph MQ[MessageQueue - Ray Actor]
        MQ_Q[deque - pickle序列化]
    end

    subgraph Trainer[Trainer - SeparateRayPPOTrainer]
        TM[FullyAsyncTrainer]
    end

    RM -->|_feed_samples| PQ[pending_queue]
    PQ -->|_processor_worker| ALM
    ALM -->|generate_sequences_single| RM
    RM -->|put_sample<br/>cloudpickle| MQ_Q
    MQ_Q -->|get_sample<br/>cloudpickle| TM
    TM -->|fit_step| TM
    RM -.->|_should_pause_generation<br/>检查queue_size| MQ_Q
    RM -.->|staleness_samples<br/>手动计数| RM
```

**问题**:

- 数据完整 `ray.cloudpickle` 序列化/反序列化开销大
- Ray Actor 单点瓶颈 (MessageQueue)
- `_should_pause_generation` 暂停逻辑复杂 (drain → resume)
- Trainer (`SeparateRayPPOTrainer`) 与 colocate 训练流程差异大，维护两套代码

### 新架构 (TQ + ReplayBuffer + PPOTrainer 多继承)

```mermaid
graph TB
    subgraph Rollouter["Rollouter (TQFullyAsyncRollouter)"]
        RM[TQFullyAsyncRollouter]
        ALM["FullyAsyncAgentLoopManager<br/>(+TQ 适配, wait=True)"]
    end

    subgraph RB["ReplayBuffer (Ray Actor)"]
        RB_SLOT["Layer1: Physical Slot<br/>acquire / release"]
        RB_VER["Layer2: Version Window<br/>reset_staleness 归零"]
        RB_META["元数据 + wait_and_sample"]
    end

    subgraph TQ[TransferQueue]
        TQ_DATA["Zero-Copy Tensor"]
    end

    subgraph Worker["AgentLoopWorkerTQ (main_ppo_sync.py)"]
        W_POST["_agent_loop_postprocess → tq.put"]
    end

    subgraph Trainer["Trainer (TQFullyAsyncTrainer)"]
        TM["多继承: PPOTrainer + FullyAsyncTrainer"]
    end

    RM -->|" 1. acquire_slot 🔒 "| RB_SLOT
    RM -->|" 2. processor "| ALM
    ALM -->|generate_sequences| Worker
    Worker -->|" 3. tq.put (status=success/finished) "| TQ_DATA
    TQ_DATA -->|kv_list poll| RB_META
    RM -->|" 4. release_slot ✅ "| RB_SLOT
    TM -->|" 5. wait_and_sample → KVBatchMeta "| RB_META
    TM -->|" 6. kv_batch_get → PPO pipeline "| TQ_DATA
    TM -->|" 7. kv_clear + remove "| TQ_DATA
    TM -->|" 8. reset_staleness "| RB_VER
```

**核心变化**:

| 维度                   | 现有架构                                           | 新架构                                         |
|----------------------|------------------------------------------------|---------------------------------------------|
| **数据通道**             | MessageQueue (pickle)                          | TransferQueue (zero-copy)                   |
| **元数据通道**            | 无 (混在数据里)                                      | ReplayBuffer (Ray Actor)                    |
| **流速控制**             | `_should_pause_generation` + staleness_samples | `acquire_slot()` 在 `_feed_samples` 源头控制     |
| **Trainer 基类**       | `SeparateRayPPOTrainer`                        | `PPOTrainer` × `FullyAsyncTrainer` 多继承      |
| **数据写入者**            | Rollouter._process_single_sample_streaming     | `AgentLoopWorkerTQ._agent_loop_postprocess` |
| **AgentLoopManager** | `FullyAsyncAgentLoopManager`                   | `FullyAsyncAgentLoopManagerTQ` (轻量子类)       |
| **暂停/恢复逻辑**          | paused + drain + resume                        | **不需要** (slot 阻塞即限流)                        |

## 核心组件

### 1. ReplayBuffer (Ray Actor) — 元数据通道 + Dual-Layer Slot 流控

文件: [`replay_buffer.py`](replay_buffer.py)

轻量级 Ray Actor，同时承担**元数据存储**和 **Dual-Layer Slot 流速控制**两大职责。

```python
@ray.remote(max_concurrency=100)
class ReplayBuffer:
    """Ray Actor: metadata channel + slot-based flow control for TQ fully async training.

    Replaces MessageQueue (data channel) in the original fully_async_policy.
    Key responsibilities:
    1. Dual-Layer slot backpressure: acquire_slot() blocks rollouter at dataloader source
    2. Metadata storage: tracks status of each sample via TQ kv_list polling
    3. Consumer interface: wait_and_sample() for trainer to get finished samples
    4. Version tracking: reset_staleness() for parameter sync coordination
    """

    def __init__(
            self,
            max_version_slots: int,  # Layer 2: 陈旧度控制
            max_pending_slots: int = 256,  # Layer 1: 物理限流
            poll_interval: float = 1.0,
    ):
```

#### Dual-Layer Slot 控制机制

`acquire_slot()` 是 Rollouter 和 RB 之间的**唯一卡控接口**，同时承担两个职责：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    acquire_slot() 双条件检查                          │
│                                                                      │
│  Layer 1: Physical (物理限流 / OOM 防护)                             │
│    条件: _pending_slots < max_pending_slots                           │
│    来源: max_concurrent_samples (如 TP×PP×16)                        │
│    作用: 防 OOM / GPU 过载                                           │
│    释放: release_slot() (Rollouter 写入 TQ 后调用)                    │
│                                                                      │
│  Layer 2: Version Window (陈旧度控制 / stale sample 防护)            │
│    条件: _version_slots < max_version_slots                           │
│    来源: required_samples × trigger_parameter_sync_step              │
│    作用: 防止样本参数版本过旧                                         │
│    释放: reset_staleness() 归零（Trainer 参数同步后调用）             │
│                                                                      │
│  ✅ 两个条件都满足 → 发放 slot (_pending_slots++, _version_slots++)  │
│  ❌ 任一不满足   → 阻塞等待                                           │
└──────────────────────────────────────────────────────────────────────┘
```

状态流转:

```mermaid
stateDiagram-v2
    [*] --> Idle: 初始化
    Idle --> Acquired: acquire_slot() 双条件通过
    Acquired --> InFlight: 写入 pending_queue
    InFlight --> Done: AgentLoopWorkerTQ 写入 TQ status=success/finished
    Done --> Idle: release_slot()
    note right of Idle
        L1: pending_slots < max_pending_slots
        L2: _version_slots < max_version_slots
    end note
    note right of Acquired
        pending_slots++
        version_slots++
        Rollouter 可以继续处理数据
    end note
    note right of InFlight
        正在生成中
        Layer1 占用 1 个物理 slot
        Layer2 占用 1 个版本槽位 (累计)
    end note
    note right of Done
        Layer1: release_slot() → pending_slots--
        Layer2: 不变 (只有 reset_staleness 归零)
    end note
```

#### 核心接口

| 接口                                                      | 调用方                                        | 说明                            |
|---------------------------------------------------------|--------------------------------------------|-------------------------------|
| `acquire_slot(timeout, uid)`                            | Rollouter._feed_samples                    | 获取写入 slot（阻塞，双条件检查）           |
| `release_slot()`                                        | Rollouter._process_single_sample_streaming | 释放物理 slot                     |
| `wait_and_sample(partition_id, sample_size, rollout_n)` | Trainer._get_keys_from_rb                  | 阻塞等待足够数量的 finish 样本           |
| `remove(partition_id, keys)`                            | Trainer._cleanup_batch                     | 移除已消费样本的元数据                   |
| `reset_staleness()`                                     | Trainer._fit_reset_staleness               | 参数同步后重置版本窗口，归零 _version_slots |
| `signal_finish()`                                       | Rollouter._streaming_generation_main       | 通知生产结束                        |

#### 后台任务

- **`_poll_from_tq()`**: 定期轮询 `tq.kv_list()` 获取 TQ 全局快照，原子替换 `self.partitions`。包含 UID 完整性检查：检测孤儿
  key（meta.uid 与 key 前缀不匹配）并自动清理。
- **`_monitor_loop()`**: 每 60 秒打印 buffer 统计信息。

#### 调用方式

```python
# Rollouter 侧 (在 Ray Actor 内部，async)
acquired = await asyncio.wrap_future(self.replay_buffer.acquire_slot.remote(timeout=None, uid=sample_id).future())

# Trainer 侧 (在 Ray Actor 内部，async)
sampled_keys_meta = await self.replay_buffer.sample.remote(
    partition_id="train", sample_size=N, rollout_n=n
)
```

---

### 2. FullyAsyncAgentLoopManagerTQ — AgentLoop 轻量适配层

文件: [`fully_async_rollouter_tq.py`](fully_async_rollouter_tq.py)

**关键点**:

- Worker 类从默认改为 `AgentLoopWorkerTQ`（定义在 `main_ppo_sync.py` 中）
- `generate_sequences_single` 增加 `wait=True`：确保 Rollouter 知道生成何时完成，避免死锁
- `AgentLoopWorkerTQ._agent_loop_postprocess` 直接将结果写入 TQ（`tq.async_kv_batch_put`），不返回数据给 Rollouter

**数据写入 TQ 后的生命周期**:

1. `AgentLoopWorkerTQ` 写入 `{uid}_{session_id}_{index}` 个 response key (status=success)
2. `AgentLoopWorkerTQ` 写入 `{uid}` uid-level key (status=finished)
3. ReplayBuffer `_poll_from_tq` 通过 `tq.kv_list()` 发现新 key，更新 `self.partitions`
4. ReplayBuffer `wait_and_sample` 检测到足够多的 finished uid，返回给 Trainer
5. Trainer 通过 `tq.kv_batch_get` 读取完整数据，执行 PPO 训练
6. Trainer 训练完成后 `tq.kv_clear` + `rb.remove` 清理

---

### 3. TQFullyAsyncRollouter — Rollouter 适配层

文件: [`fully_async_rollouter_tq.py`](fully_async_rollouter_tq.py)

基于 `FullyAsyncRollouter` 的增量修改子类。核心变化集中在数据馈送、样本处理和验证三个环节。

#### 3.1 `_feed_samples` — 源头限流

```python
async def _feed_samples(self):
    """Feed samples from dataloader to pending_queue, with source-level flow control.

    Key difference from base class: acquire_slot() is called BEFORE putting
    to pending_queue. This blocks the dataloader when too many samples are
    in-flight, replacing the need for _should_pause_generation().
    """
    continuous_iterator = self._create_continuous_iterator()
    rollout_n = self.config.actor_rollout_ref.rollout.n

    for epoch, batch_dict in continuous_iterator:
        sample_id = f"sample_{epoch}_{self.global_steps}"
        # ★ 核心: 在获取数据后立即申请 slot（可能阻塞）
        acquired = await self.replay_buffer.acquire_slot.remote(timeout=None, uid=sample_id)
        if not acquired:
            break

        # 注入元数据字段到 batch_dict (plain dict)，在 tu.get_tensordict() 之前
        # np.array 值会通过 get_tensordict 转为 NonTensorStack，支持按索引访问
        batch_dict["uid"] = np.array([sample_id], dtype=object)
        batch_dict["__rollout_n__"] = np.full(1, rollout_n, dtype=np.int64)
        batch_dict["sample_id"] = np.array([sample_id], dtype=object)
        batch_dict["global_steps"] = np.full(1, self.global_steps, dtype=np.int64)

        full_batch = tu.get_tensordict(batch_dict)
        # ... 构建 RolloutSample 并放入 pending_queue ...
        await self.pending_queue.put(rollout_sample)

    # 结束信号
    await self.pending_queue.put(None)
```

**与基类的关键区别**:

- 不再调用 `prepare_single_generation_data()` (不做 repeat(n))，改为注入 `__rollout_n__` 字段让
  `AgentLoopWorkerTQ._run_prompt` 内部循环 n 次
- `batch_size=1` (bsz=1)，每个 prompt 单独处理
- `uid`/`__rollout_n__`/`sample_id`/`global_steps` 作为 `np.array` 注入到 plain dict 中，在 `tu.get_tensordict()` 后变为
  `NonTensorStack`

#### 3.2 `_process_single_sample_streaming` — 简化为 generate + release

```python
async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
    """Process a single sample: generate via ALM worker (which writes to TQ blocking),
    then release slot.

    Simplified from base class:
    - Base class: generate → put to MessageQueue
    - TQ path: generate + TQ write both happen INSIDE AgentLoopWorkerTQ
      (via overridden _agent_loop_postprocess)
      → we just call it and release the slot
    """
    try:
        # 调用 ALM (内部走 AgentLoopWorkerTQ → 写入 TQ)
        await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        self.total_generated_samples += 1
    except Exception as e:
        logger.exception(f"Failed to process {rollout_sample.sample_id}: {e}")
    finally:
        # ★ 无论成功失败都释放 slot
        await self.replay_buffer.release_slot.remote()

    self.processed_sample_count += 1
```

**与基类的核心区别**: 基类需要手动将生成结果 put 到 MessageQueue；TQ 路径下数据写入由
`AgentLoopWorkerTQ._agent_loop_postprocess` 完成，Rollouter 只需调用 generate 然后 release_slot。

#### 3.3 删除/禁用的方法

| 方法                           | 处理方式       | 原因                                |
|------------------------------|------------|-----------------------------------|
| `_should_pause_generation()` | 返回 `False` | 由 `acquire_slot` 替代               |
| `_async_monitor_loop()`      | 空实现        | 监控由 ReplayBuffer._monitor_loop 承担 |

#### 3.4 验证流程 `_validate`

覆盖基类的验证方法，使用 TQ + ReplayBuffer 路径：

```python
async def _validate(self) -> dict[str, float]:
    for batch_dict in self.val_dataloader:
        # 1. dispatch to agent loop manager (writes to TQ via AgentLoopWorkerTQ)
        batch = tu.get_tensordict(batch_dict)
        self.async_rollout_manager.generate_sequences(batch)

        # 2. sample batch from replay buffer (blocks until enough finish samples)
        batch = await self.replay_buffer.sample(
            partition_id="val", sample_size=len(batch),
            rollout_n=self.config.actor_rollout_ref.rollout.val_kwargs.n
        ).remote()

        # 3. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            batch = self._compute_reward_colocate(batch)

        # 4. read text data from TQ
        text_data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                    select_fields=["prompts", "responses"])

        # 5. cleanup
        tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
        self.replay_buffer.remove(batch.partition_id, batch.keys)

    return self._val_metrics_update(...)
```

---

### 4. TQFullyAsyncTrainer — 多继承 Trainer

文件: [`fully_async_trainer_tq.py`](fully_async_trainer_tq.py)

**最核心的设计决策**: 通过 Python 多继承 `class FullyAsyncTrainerTQ(PPOTrainer, FullyAsyncTrainer)` 同时获得两边能力：

```python
"""
MRO: TQFullyAsyncTrainer → PPOTrainer → FullyAsyncTrainer → SeparateRayPPOTrainer → ...

Data flow:
    TQFullyAsyncRollouter --(tq.kv_batch_put)--> TransferQueue (status=finish)
        |
    TQFullyAsyncTrainer <-(RB.wait_and_sample)--+--(KVBatchMeta)--> [PPOTrainer pipeline]
                                                    |
                                              update_actor(KVBatchMeta)
"""
```

#### 4.1 `__init__` — 双路径初始化

```python
class FullyAsyncTrainerTQ(PPOTrainer, FullyAsyncTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping,
                 resource_pool_manager, device_name=None, ...):
        # ======== 1. PPOTrainer.__init__: config, dataloader, local replay_buffer, worker groups ========
        PPOTrainer.__init__(self, config=config, role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager)
        # 手动设置 PPOTrainer 不初始化但 FullyAsyncTrainer 需要的字段
        self.device_name = device_name
        self.ray_worker_group_cls = ray_worker_group_cls or RayWorkerGroup
        self.tokenizer = tokenizer

        # ======== 2. FullyAsyncTrainer state fields ========
        self.global_steps = 0
        self.current_param_version = 0
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        # ... (mirrors FullyAsyncTrainer.__init__ lines 108-163)

        # ======== 3. TQ-specific ========
        self.replay_buffer = None  # Set via set_replay_buffer()
```

#### 4.2 `fit()` — 异步训练主循环

```python
async def fit(self):
    """Main training loop: async RB consumption + PPOTrainer step() pipeline."""
    self.global_steps += 1

    while True:
        try:
            await self.fit_step()
        except TrainingStopException:
            break

    # 收尾: 最终 validate + checkpoint
    if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
        await self._fit_update_weights()
        await self._fit_validate()
    self._fit_save_checkpoint(force=True)
```

#### 4.3 `fit_step()` — 单步训练

这是整个 TQ 路径的核心：**完全复用 PPOTrainer 的 _compute_* / _update_* 方法**，仅替换数据来源：`

#### 4.4 `_get_keys_from_rb()` — 从 RB 获取 KVBatchMeta

返回的 `KVBatchMeta` 可直接传入所有 PPOTrainer 的 `_compute_*` 方法——这些方法内部通过 `tq.kv_batch_get` 从 TQ 读取实际张量数据。
---

## 数据流详解

### 完整生命周期时序图

```mermaid
sequenceDiagram
    participant R as Rollouter<br/>(_feed_samples)
    participant RB as ReplayBuffer<br/>(Ray Actor)
    participant PQ as pending_queue
    participant P as processor<br/>(_processor_worker)
    participant ALM_TQ as ALM_TQ<br/>(DataProto→TensorDict)
    participant ALW as AgentLoopWorkerTQ<br/>(main_ppo_sync.py)
    participant TQ as TransferQueue
    participant T as Trainer<br/>(TQFullyAsyncTrainer)
    Note over R, T: === Phase 1: Rollouter 生产 (源头限流) ===
    loop dataloader iteration
        R ->> RB: acquire_slot(uid=sample_id) [可能阻塞🔒]
        RB -->> R: slot acquired (L1++, L2++)
        R ->> R: 注入 uid/__rollout_n__/sample_id/global_steps
        R ->> R: tu.get_tensordict(batch_dict) → bsz=1 TensorDict
        R ->> PQ: put(RolloutSample{full_batch, sample_id})
    end
    R ->> PQ: put(None) [结束信号]
    Note over R, T: === Phase 2: 生成 (ALM_TQ 适配 + Worker 写 TQ) ===
    loop processor worker
        P ->> PQ: get()
        P ->> ALM_TQ: generate_sequences_single(batch, wait=True)
        ALM_TQ ->> ALM_TQ: 选择 worker (round-robin)
        ALM_TQ ->> ALW: generate_sequences.remote(batch, wait=True)
        ALW ->> ALW: _run_prompt 循环 __rollout_n__ 次
        ALW ->> ALW: _compute_score (reward model)
        ALW ->> TQ: tq.async_kv_batch_put(response keys, status=success)
        ALW ->> TQ: tq.async_kv_batch_put(uid key, status=finished)
        ALW -->> ALM_TQ: 返回 (wait=True 确保完成)
        ALM_TQ -->> P: 返回
        P ->> RB: release_slot() ✅ (L1--)
        RB -->> R: (唤醒等待的 acquire_slot)
    end

    Note over R, T: === Phase 3: Trainer 消费 + PPO 训练 ===
    loop fit_step (每步)
        T ->> RB: wait_and_sample(sample_size=N, rollout_n=n) [阻塞⏳]
        RB -->> T: [(key1,tag1), ..., (keyN,tagN)] (KVBatchMeta)
        T ->> TQ: kv_batch_get(keys) [内部各 _compute_* 方法调用]
        TQ -->> T: 完整张量数据 (prompts, responses, log_probs, ...)
        T ->> T: _balance_batch → _compute_old_log_prob
        T ->> TQ: kv_batch_put(old_log_prob, ...) [写回 TQ]
        T ->> T: _compute_advantage → _update_actor
        T ->> TQ: kv_batch_put(advantages, returns, ...)
        T ->> TQ: kv_clear(response keys + uid keys)
        T ->> RB: remove(keys)
    end

    Note over R, T: === Phase 4: 参数同步 (每 trigger_parameter_sync_step 步) ===
    T ->> T: _fit_update_weights(): checkpoint_manager.update_weights()
    T ->> RB: reset_staleness(): _version_slots 归零, 通知 L2 解除阻塞
```

### Dual-Layer Slot Control 详细语义

| 原概念 (MessageQueue 路径)         | 新实现 (TQ 路径)                         |
|-------------------------------|-------------------------------------|
| `MessageQueue.queue_size`     | `RB._pending_slots` (Layer 1: 物理限流) |
| `max_queue_size`              | `max_pending_slots` (Layer 1)       |
| `_should_pause_generation()`  | **删除** — `acquire_slot()` 双条件即限流    |
| `staleness_samples` (手动计数)    | `RB._version_slots` (Layer 2: 累计计数) |
| `max_required_samples`        | `max_version_slots` (Layer 2)       |
| `paused` + `drain` + `resume` | **保留但不再触发卡控** — 仅用于参数同步时的安全 drain   |

**改动前后的对比**:

```
改动前 (双重限流, 复杂状态机):
  1. _should_pause_generation(): queue_size >= max_queue_size → pause
  2. _should_pause_generation(): staleness_samples >= max_required_samples → pause
  → 需要 paused/drain/resume 状态机

改动后 (acquire_slot 单接口双条件):
  1. _feed_samples(): acquire_slot()
     → Layer 1 不满足? 阻塞 (物理满)
     → Layer 2 不满足? 阻塞 (版本窗口满, 等 reset_staleness)
     → 都满足? 放行
  → 简洁的令牌桶语义, 无需额外状态机
```

**参数同步流程**:

```
Trainer.fit_step:
  1. wait_and_sample(batch_size) → 从 RB 获取 finish 样本
  2. ... PPO 训练流程 (_compute_* / _update_*) ...
  3. update_weights() → NCCL 同步权重到 Rollouter GPUs
  4. reset_staleness():
       a. _version_slots = _pending_slots + train_finished_slots (重新计算)
       b. 重置计时器 (step_start_time, idle_start_time)
       c. 通知 _slot_available → 解除 acquire_slot 的 Layer 2 阻塞
```

## 使用方法

### 启动脚本示例

核心配置要点:

```bash
# ====== 必要的 fully_async 配置 ======
fully_async=(
  data.train_batch_size=0                 # TQ 模式下无效，默认 0
  data.gen_batch_size=1                   # streaming 逐条生成
  trainer.test_freq=-1                     # 由 rollouter 负责 validate
  actor_rollout_ref.hybrid_engine=False    # 分离式架构
  actor_rollout_ref.rollout.calculate_log_probs=True  # 使用 rollout log_prob
  rollout.total_rollout_steps=$(((512*100)))          # 总生成样本数
  trainer.nnodes=1                         # Trainer 节点数
  trainer.n_gpus_per_node=4                # Trainer 每 GPU 数
  rollout.nnodes=1                         # Rollouter 节点数
  rollout.n_gpus_per_node=4                # Rollouter 每 GPU 数
  async_training.staleness_threshold=0.5   # 陈旧度阈值
  async_training.trigger_parameter_sync_step=4  # 参数同步频率
  async_training.require_batches=1         # 每次 fetch 的 batch 数
  async_training.partial_rollout=True       # 支持 partial rollout
)

# ====== TQ 特有配置 ======
transfer_queue=(
  transfer_queue.enable=True               # ★ 启用 TQ 模式
)
```

### 依赖安装

```bash
pip install TransferQueue==0.1.6
```

所有 TQ 相关代码都有 fallback：当 `import transfer_queue` 失败时，自动使用 `verl.utils.transferqueue_utils` 中的 mock 实现。