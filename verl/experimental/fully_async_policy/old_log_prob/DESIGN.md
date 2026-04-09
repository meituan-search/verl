# old_log_prob 设计文档

## 概述

`old_log_prob` 是一个独立的推理服务，用于在 fully-async PPO 训练中实时计算历史 log 概率（old log probabilities）。它在 rollout 生成期间与 vLLM/SGLang 并行运行，使用与 Trainer 权重同步的 Megatron/FSDP 训练引擎执行前向推理。

## 组件说明

### 类层次

```
RolloutReplica (ABC)
└── OldLogProbReplica          # 副本生命周期管理，与 vLLM/SGLang Replica 对称

CheckpointEngineWorker (Worker)
└── OldLogProbWorker           # GPU 算子执行单元，参与 NCCL 权重同步

BaseRollout (ABC)
└── OldLogProbServerAdapter    # NCCL 权重接收 → CPU 暂存 → TrainingWorker 写入

OldLogProbServer (Ray Actor)   # 请求批量调度器，实现 abort/resume 协议
```

### 文件位置

| 组件 | 文件 |
|------|------|
| `OldLogProbServerAdapter` | `verl/workers/old_log_prob/old_log_prob.py` |
| `OldLogProbWorker` | `verl/workers/old_log_prob/old_log_prob.py` |
| `OldLogProbServer` | `verl/workers/old_log_prob/old_log_prob.py` |
| `OldLogProbReplica` | `verl/workers/old_log_prob/old_log_prob.py` |
| 初始化入口 | `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` |
| 权重同步入口 | `verl/checkpoint_engine/base.py` (`CheckpointEngineManager`) |

---

## 一、初始化流程

```
fully_async_main._initialize_components()
  │
  ├─ trainer.init_workers()
  │    └─ FullyAsyncTrainer._init_models()
  │         └─ actor_wg.init_model()  (Trainer 侧，不涉及 old_log_prob)
  │
  └─ rollouter.init_workers()
       └─ FullyAsyncRollouter._init_async_rollout_manager()
            └─ FullyAsyncAgentLoopManager.create()
                 └─ _initialize_llm_servers()
                      ├─ super()._initialize_llm_servers()   # vLLM/SGLang replicas
                      └─ if config.old_log_prob.enable_standalone:
                           _init_old_log_prob_replica()
                             │
                             ├─ OldLogProbReplica(replica_rank, full_config, OldLogProbWorker)
                             │    └─ __init__: 从 full_config.old_log_prob 读取
                             │                n_gpus_per_node, nnodes
                             │                直接设置 world_size/nnodes 等字段
                             │
                             └─ replica.init_standalone()
                                  ├─ ResourcePoolManager.create_resource_pool()
                                  │    └─ Ray placement group 分配
                                  │
                                  ├─ RayWorkerGroup(resource_pool, OldLogProbWorker)
                                  │    └─ 每个 GPU 启动一个 OldLogProbWorker Ray Actor
                                  │
                                  └─ launch_servers()
                                       ├─ worker_group.init_model()
                                       │    └─ OldLogProbWorker.init_model() [ONE_TO_ALL]
                                       │         └─ OldLogProbServerAdapter.init_model()
                                       │              ├─ 重映射 config key
                                       │              │  (micro_batch_size → ppo_*)
                                       │              ├─ TrainingWorker(TrainingWorkerConfig)
                                       │              └─ training_worker.reset() (加载模型权重)
                                       │
                                       └─ OldLogProbServer.remote(worker_group, cfg)
                                            ├─ 启动 _batch_consumer Task
                                            └─ self.servers = [server]

                 └─ _init_agent_loop_workers()
                      └─ FullyAsyncAgentLoopWorker.remote(..., old_log_prob_server_handle)
                           └─ FullyAsyncLLMServerManager(old_log_prob_server_handle=server)
```

**关键设计**：
- `OldLogProbReplica.init_standalone()` 自建独立 Ray resource pool，与 vLLM/SGLang Replica 完全对称
- `init_model()` 延迟到 Ray Actor 启动后执行，避免在 `__init__` 时占用 GPU 内存
- `OldLogProbReplica` 追加到 `rollout_replicas`，使 Trainer 的 `CheckpointEngineManager` 通过 `rollouter.get_replicas()` 自动获取并管理其权重同步

---

## 二、计算流程（Rollout 期间）

```
FullyAsyncAgentLoopWorker.generate_sequences(prompts)
  └─ server_manager.generate(request_id, prompt_ids, sampling_params)
       └─ FullyAsyncLLMServerManager.generate()
            │
            └─ while True:  (partial rollout 循环)
                 ├─ super().generate()  → vLLM/SGLang 生成 tokens
                 │
                 └─ if not validate:
                      _recompute_old_log_prob(output, prompt_ids, temperature)
                        │
                        ├─ if old_log_prob_server_handle is None: return  (跳过)
                        │
                        ├─ 构造 TensorDict:
                        │    prompts (left-padded)  + responses (right-padded)
                        │    input_ids, attention_mask, position_ids, response_mask
                        │    temperature, max_response_len
                        │
                        └─ await server_handle.compute_old_log_prob.remote(data_td)
                               │
                               └─ OldLogProbServer.compute_old_log_prob(data_td)
                                    ├─ _drain_done.wait()  (等待未在更新权重)
                                    ├─ 入队 _request_queue
                                    └─ await future  (等待批量推理结果)

OldLogProbServer._batch_consumer (后台 Task)
  └─ 收集 batch_size 个请求 (或 timeout 触发)
       └─ _execute_batch(requests, futures)
            └─ async with _infer_lock:
                 infer_batch(batched_data)
                   ├─ 补 padding 至 _min_dispatch_unit 的整数倍
                   ├─ left_right_2_no_padding(data)
                   └─ worker_group.compute_log_prob(data)  [ND_COMPUTE]
                        └─ OldLogProbWorker.compute_log_prob(data)  [同步]
                             └─ OldLogProbServerAdapter.compute_log_prob(data)  [同步]
                                  └─ training_worker.infer_batch(data)
                                       → TrainingWorker 前向推理
                                       → 返回 log_probs
```

**关键设计**：
- `_drain_done` Event 门控请求入队，权重更新期间阻止新请求
- `_batch_consumer` 后台 Task 持续聚合请求，支持 `batch_size` 满触发和 `timeout` 超时触发两种分发策略
- `_infer_lock` 序列化 GPU 推理，防止推理与权重加载并发
- `_min_dispatch_unit = dp_size * micro_batch_size_per_gpu`，确保每次分发的 batch 大小是 DP 并行的整数倍

---

## 三、更新权重流程

```
FullyAsyncTrainer._fit_update_weights()
  └─ checkpoint_manager.update_weights(global_steps)
       │
       │  replicas = rollouter.get_replicas()
       │           = async_rollout_manager.rollout_replicas
       │           = [vLLMReplica(s), ..., OldLogProbReplica]
       │
       ├─ ① for replica in replicas:
       │       replica.abort_all_requests()
       │         └─ vLLM/SGLang: pause_generation()，保存 in-flight 请求
       │         └─ OldLogProbReplica: no-op
       │              ↑ 此窗口期：被 vLLM/SGLang partial 打断的样本
       │                仍可向 OldLogProbServer 发送请求，用当前权重计算 log prob
       │
       ├─ ② for replica in replicas:
       │       replica.sleep()
       │         └─ vLLM/SGLang: 释放 KV cache GPU 内存
       │         └─ OldLogProbReplica → server.drain_and_lock.remote()
       │              └─ OldLogProbServer.drain_and_lock()
       │                   ├─ _drain_done.clear()      ← 阻止新请求入队
       │                   ├─ _infer_lock.acquire()    ← 等当前推理批次完成
       │                   ├─ 启动 lock_holder_task    ← 持锁跨越 sleep/resume 边界
       │                   └─ flush 队列剩余请求        ← 用旧权重处理完
       │
       ├─ ③ build_process_group(rollout_wg)
       │       │
       │       │  # 将所有 replica.workers 合并为一个临时 RayWorkerGroup：
       │       │  workers = []
       │       │  for replica in replicas:
       │       │      workers.extend(replica.workers)
       │       │  # workers = [vLLMWorker×N, ..., OldLogProbWorker×M]
       │       │  rollout = RayWorkerGroup(worker_handles=workers,
       │       │                ray_cls_with_init=RayClassWithInitArgs(cls=CheckpointEngineWorker))
       │       │  # OldLogProbWorker 继承 CheckpointEngineWorker，方法接口完全兼容
       │       │
       │       ├─ worker.execute_checkpoint_engine("prepare")
       │       │    └─ OldLogProbWorker.execute_checkpoint_engine [DP_COMPUTE]
       │       │         → checkpoint_engine.prepare()  (分配 NCCL buffer)
       │       ├─ build_topology()  → 计算 NCCL rank 拓扑
       │       └─ worker.execute_checkpoint_engine("init_process_group")
       │            └─ OldLogProbWorker 被分配独立的 NCCL rank，加入统一通信组
       │
       ├─ ④ trainer.update_weights() + rollout.update_weights()
       │       └─ OldLogProbWorker.update_weights() [ONE_TO_ALL, blocking=False]
       │            └─ checkpoint_engine.receive_weights()  ← NCCL 接收权重流
       │            └─ server_adapter.update_weights(weights)
       │                 └─ OldLogProbServerAdapter.update_weights()
       │                      └─ async for name, tensor in weights:
       │                             staged[name] = tensor.to("cpu")
       │                         cuda.synchronize()
       │                         _staged_state_dict = staged  ← CPU 暂存
       │
       ├─ ⑤ worker.execute_checkpoint_engine("finalize")
       │       → destroy NCCL group, free buffer
       │
       ├─ ⑥ for replica in replicas:
       │       replica.wake_up()
       │         └─ vLLM/SGLang: 恢复 KV cache GPU 内存
       │         └─ OldLogProbReplica → server.load_weights_and_unlock.remote()
       │              └─ OldLogProbServer.load_weights_and_unlock()
       │                   ├─ _resume_event.set()          ← 释放 _infer_lock
       │                   ├─ worker_group.load_staged_weights()
       │                   │    └─ OldLogProbWorker.load_staged_weights() [ONE_TO_ALL]
       │                   │         └─ OldLogProbServerAdapter.load_staged_weights()
       │                   │              └─ engine.set_param(staged_state_dict)
       │                   │                   ← 一次性写入 GPU
       │                   └─ _drain_done.set()            ← 开放请求入队
       │
       └─ ⑦ for replica in replicas:
               replica.resume_generation()
                 └─ vLLM/SGLang: 恢复 in-flight 请求（新权重）
                 └─ OldLogProbReplica: no-op
```

**关键设计**：
- **统一 worker group**：`CheckpointEngineManager` 把所有 replica（vLLM/SGLang/OldLogProb）的 `workers` 合并为一个临时 `RayWorkerGroup`，以 `CheckpointEngineWorker` 为接口类型统一调度。`OldLogProbWorker` 继承 `CheckpointEngineWorker`，接口完全兼容，无需特殊处理
- **两步加载**：NCCL 接收 → CPU 暂存（`update_weights`）→ `wake_up` 时 `set_param()` 一次性写入 GPU。这是因为 `TrainingWorker.engine.set_param()` 需要完整 `state_dict`，不支持流式热更新（vLLM/SGLang 支持流式热更新，不需要此设计）
- **partial rollout 窗口期**：`abort_all_requests` 对 old_log_prob 是 no-op，使 vLLM/SGLang 被打断的样本在 ①→② 之间仍可向 OldLogProbServer 发请求，用当前权重完成 log prob 计算
- **锁跨越 sleep/wake_up 边界**：`drain_and_lock()` 在 `sleep()` 中调用，`lock_holder_task` 持有 `_infer_lock` 直到 `wake_up()` 中的 `load_weights_and_unlock()` 调用，确保 NCCL 传输（步骤 ④）期间绝无推理发生，权重不会被混用
- `OldLogProbReplica` 在 `rollout_replicas` 中，`CheckpointEngineManager` 统一管理其生命周期

---

## 四、结束流程

```
FullyAsyncRollouter.fit()
  └─ finally:
       └─ async_rollout_manager.shutdown()
            └─ FullyAsyncAgentLoopManager.shutdown()
                 └─ for replica in rollout_replicas:
                      if hasattr(replica, "shutdown"):
                          await replica.shutdown()
                            └─ OldLogProbReplica.shutdown()
                                 └─ server.shutdown.remote()
                                      └─ OldLogProbServer.shutdown()
                                           ├─ _shutdown = True
                                           ├─ _consumer_task.cancel()
                                           └─ 清空 _request_queue
                                                (pending futures set exception)
```

**关键设计**：
- `OldLogProbServer` 生命周期完全归属 Rollouter 侧，由 `FullyAsyncRollouter.fit()` 的 `finally` 块触发关闭
- Trainer 侧完全不感知 old_log_prob 的存在

---

## 五、OldLogProbServer 内部架构

### 状态机

`OldLogProbServer` 围绕三个 asyncio 原语维护状态：

| 原语 | 类型 | 初始状态 | 作用 |
|------|------|---------|------|
| `_drain_done` | `asyncio.Event` | set（开放） | 控制请求是否可以入队 |
| `_infer_lock` | `asyncio.Lock` | unlocked | 序列化推理执行，防止推理与权重加载并发 |
| `_resume_event` | `asyncio.Event` | clear | 通知 `lock_holder_task` 释放 `_infer_lock` |

### 并发组件

```
OldLogProbServer（Ray Actor，单线程 asyncio 事件循环）
│
├─ _batch_consumer Task（后台持续运行）
│    └─ 等待 _drain_done → 收集请求 → _execute_batch()
│         └─ async with _infer_lock → run_in_executor(infer_batch)
│
├─ _lock_holder_task（仅在 drain_and_lock 期间存在）
│    └─ 等待 _resume_event → 释放 _infer_lock
│
└─ compute_old_log_prob() / drain_and_lock() / load_weights_and_unlock()
     （由外部 Ray remote call 触发）
```

### 批量收集算法（_batch_consumer）

```
while not shutdown:
  await _drain_done.wait()          # 等待请求门开放
  deadline = now + timeout
  batch = []

  while len(batch) < batch_size:
    remaining = deadline - now
    if remaining <= 0: break         # 超时，分发当前批次

    item = await queue.get(timeout=remaining)

    if _drain_done cleared:          # drain 在收集中途触发
      put items back to queue        # 将已收集项目归还
      goto top of outer loop         # 等待 drain 完成后重新开始

    batch.append(item)
    if len(batch) == batch_size: break  # 批次满，立即分发

  if batch:
    await _execute_batch(batch)
```

**关键设计**：
- `batch_size` 满触发和 `timeout` 超时触发两种分发策略，平衡延迟与吞吐
- drain 在收集中途触发时，已收集的请求**归还队列**而非丢弃，保证请求不丢失
- `_execute_batch` 持有 `_infer_lock`，与 `drain_and_lock` 互斥，确保不会在权重传输时执行推理

### infer_batch 数据处理流程

```
输入: TensorDict (batch_size 个请求，每个 batch_size[0]=1)
  │
  ├─ 补 padding 至 _min_dispatch_unit 的整数倍
  │    _min_dispatch_unit = dp_size * micro_batch_size_per_gpu
  │    （use_dynamic_bsz 时 = dp_size）
  │
  ├─ left_right_2_no_padding()   ← 去掉 padding，提升计算效率
  ├─ assign_non_tensor(calculate_entropy=True, compute_loss=False)
  │
  ├─ worker_group.compute_log_prob(data)  ← 分发到 OldLogProbWorker（同步）
  │    └─ OldLogProbWorker.compute_log_prob(data)  [同步，blocking=True]
  │         └─ OldLogProbServerAdapter.compute_log_prob(data)  [同步]
  │              └─ training_worker.infer_batch(data)
  │
  ├─ no_padding_2_padding(log_probs)      ← 还原 padding 格式
  └─ log_probs[:n_real]                   ← 丢弃 padding dummy 行
```

**注**：`compute_log_prob` 为同步方法。`OldLogProbServer.infer_batch` 通过 `run_in_executor` 在线程池中执行，不阻塞 asyncio 事件循环。`_infer_lock` 在 `_execute_batch` 中持有，确保推理与权重传输不并发，无需额外的 `_gpu_lock`。

### 权重更新期间的锁协议

```
正常推理（_batch_consumer）:
  _drain_done.is_set()  →  请求入队  →  _execute_batch  →  async with _infer_lock

drain_and_lock()（sleep 时调用）:
  _drain_done.clear()          # 阻止新请求入队
  _infer_lock.acquire()        # 等待当前批次完成
  _lock_holder_task = Task(_hold_infer_lock_until_resume)
                               # 后台任务持锁，等待 _resume_event
  flush 队列剩余请求            # 用旧权重处理完

  ← NCCL 权重传输（此时 _infer_lock 被持有，推理完全阻断）

load_weights_and_unlock()（wake_up 时调用）:
  _resume_event.set()          # 通知 lock_holder_task 释放锁
  await _lock_holder_task      # 等待锁真正释放
  worker_group.load_staged_weights()  # 写入新权重
  _drain_done.set()            # 开放请求入队
```

**锁持有时间线**：

```
drain_and_lock()  │←────── _infer_lock 持有 ──────→│  load_weights_and_unlock()
                  │                                  │
                  ↓                                  ↓
         sleep()调用                          wake_up()调用
                  │←── NCCL 传输 ──→│←── finalize ──→│
```

---

## 六、四个流程的关键设计要点

| 流程 | 关键机制 |
|------|---------|
| **初始化** | `OldLogProbReplica.init_standalone()` 自建 resource pool，与 vLLM/SGLang 完全对称；`init_model()` 延迟到 actor 启动后执行 |
| **计算** | `_drain_done` Event 门控请求入队；`_batch_consumer` 后台 Task 聚合批次；`_infer_lock` 序列化推理（`run_in_executor` 中执行）；`compute_log_prob` 为同步方法，无 `_gpu_lock` |
| **更新权重** | 两步加载：NCCL 接收 → CPU 暂存；`wake_up` 时 `set_param()` 一次性写入；`_infer_lock` 跨越 sleep/wake_up 边界确保无混合权重推理 |
| **结束** | 由 Rollouter 侧 `fit()` 的 `finally` 块触发，生命周期完全归属 Rollouter |

---

## 六、与 vLLM/SGLang 的对比

| 方面 | vLLM/SGLang | old_log_prob |
|------|-------------|-------------|
| 推理引擎 | 专用 serving 引擎（AsyncLLM / SGLang） | TrainingWorker（Megatron/FSDP） |
| 权重接收 → 写入 | 流式热更新（直接写入引擎） | 两步：CPU 暂存 → `set_param()` |
| 请求调度 | 引擎内置调度器 | `OldLogProbServer` 手动 batch 队列 |
| abort 行为 | `pause_generation()`，保存请求状态 | 停止入队 + 持 `_infer_lock` + 刷队列 |
| resume 行为 | `resume_generation()`，恢复请求 | 释放锁 + 加载权重 + 开放入队 |
| 所属侧 | Rollouter 侧 | Rollouter 侧 |
| Replica 类 | `vLLMReplica` / `SGLangReplica` | `OldLogProbReplica` |
