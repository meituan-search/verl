# 弹性 Rollout 与训练调度系统

## 1. 整体架构

### 1.1 资源划分模型

弹性调度系统将 GPU 集群划分为**两类**资源：

- **固定 Rollout 资源（Fixed Rollout）**：专用于推理生成，持续产生样本，由独立的 vLLM/SGLang 服务承载，始终保持活跃
- **弹性资源（Elastic）**：可在 Rollout 和 Train 两种角色之间动态切换，由 `ElasticActorWorker` 承载

> **没有"固定 Train 资源"**。所有训练算力均来自弹性资源。可通过 `min_train_resources` 参数保留至少 N 个弹性单元始终处于 Train 模式，以保证训练不中断。

```mermaid
graph TB
    subgraph cluster["GPU Cluster"]
        subgraph fixed["固定 Rollout 资源（始终活跃）"]
            FR["vLLM / SGLang Server<br/>DP0, DP1, ..."]
        end
        subgraph elastic["弹性资源（动态切换）"]
            EW0["ElasticActorWorker-0<br/>当前: ROLLOUT 或 TRAIN"]
            EW1["ElasticActorWorker-1<br/>当前: ROLLOUT 或 TRAIN"]
            EWN["ElasticActorWorker-N<br/>..."]
        end
    end

    subgraph control["调度控制层"]
        EC["ElasticCoordinator<br/>监控速率 · 决策切换 · 延迟到训练边界"]
    end

    subgraph app["应用层"]
        ER["ElasticRollouter<br/>样本生产"]
        ET["ElasticTrainer<br/>模型训练"]
        MQ["MessageQueue<br/>样本缓冲"]
    end

    FR -->|固定推理流| ER
    EW0 -->|Rollout 模式| ER
    EW1 -->|Rollout 模式| ER
    EW0 -->|Train 模式| ET
    EW1 -->|Train 模式| ET
    ER --> MQ
    MQ --> ET
    EC -->|switch_elastic_to_rollout / switch_elastic_to_train| ET
    ET -->|统计消费速率| EC
    ER -->|统计生产速率| EC
    MQ -->|队列利用率| EC
```

### 1.2 弹性资源双模式设计（核心）

每个 `ElasticActorWorker` 同时持有 **Actor Engine**（训练引擎）和 **Rollout Engine**（推理服务器），两者互斥地占用同一份 GPU 显存：

```mermaid
flowchart LR
    INIT(["init_model()"]) --> TRAIN

    TRAIN(["🟢 TRAIN\nActor Engine 在 GPU\nRollout Engine 休眠"])
    ROLLOUT(["🔵 ROLLOUT\nRollout Engine 在 GPU\nActor Engine 在 CPU"])

    TRAIN -->|"switch_elastic_to_rollout()"| T2R
    T2R -->|完成| ROLLOUT

    ROLLOUT -->|"switch_elastic_to_train()"| R2T
    R2T -->|完成| TRAIN

    subgraph T2R ["Train → Rollout 切换步骤"]
        direction TB
        s1["① remove_elastic_actor()\nDP rebuild 排除本 rank"]
        s2["② actor.offload_to_cpu()\n释放 GPU 显存"]
        s3["③ rollout.wake_up()\n加入 LB 池开始生产"]
        s1 --> s2 --> s3
    end

    subgraph R2T ["Rollout → Train 切换步骤"]
        direction TB
        t1["① rollout.sleep() + abort\n释放 GPU 显存"]
        t2["② actor.load_to_gpu()\n恢复训练权重"]
        t3["③ add_elastic_actor()\nDP rebuild 纳入本 rank"]
        t1 --> t2 --> t3
    end
```

**GPU 显存安全保证**：
- **Train → Rollout**：Actor 权重先 offload 到 CPU，再 wake_up rollout server
- **Rollout → Train**：Rollout server 先 sleep 释放显存，再 load actor 权重到 GPU

### 1.3 调度时序

```mermaid
sequenceDiagram
    participant EC as ElasticCoordinator
    participant ET as ElasticTrainer
    participant EW as ElasticActorWorker
    participant ER as ElasticRollouter
    participant MQ as MessageQueue
    
    Note over EC: 后台监控循环（每 check_interval 秒）
    loop 持续监控
        EC->>MQ: 获取队列利用率
        EC->>ER: 获取生产速率
        EC->>ET: 获取消费速率
        EC->>EC: 评估拥塞，写 _pending_action flag
    end
    
    Note over ET: 每个 fit_step 训练前边界
    ET->>MQ: _fit_generate() — 等待收满 required_samples
    Note over ET: Train GPU 完全空闲

    ET->>EC: on_before_fit_step()
    EC->>EC: 检查 _pending_action（含冷却时间校验）

    alt scale_rollout（队列过空，生产跟不上消费）
        EC->>ET: switch_elastic_to_rollout(resource_id)
        ET->>ET: remove_elastic_actor() — DP rebuild 排除此 rank
        ET->>EW: switch_to_rollout() — offload actor 到 CPU
        ET->>ER: add_elastic_replica() — wake_up server，注册 LB
    else scale_train（队列过满，消费跟不上生产）
        EC->>ET: switch_elastic_to_train(resource_id)
        ET->>ER: remove_elastic_replica() — sleep server，abort in-flight
        ET->>EW: switch_to_train() — load actor 到 GPU
        ET->>ET: add_elastic_actor() — DP rebuild 纳入此 rank
    end
        
    Note over ET: 切换完成，GPU 仍空闲，开始训练计算
    ET->>ET: _fit_compute_reward() / _fit_update_actor() ...
```
        
---
    
## 2. 核心组件
    
### 2.1 ElasticCoordinator
    
`ElasticCoordinator` 是系统的调度大脑，以 Ray Actor 形式运行。
    
**职责**：
1. 后台轮询 MQ 队列利用率、Rollouter 生产速率、Trainer 消费速率
2. 基于 EMA 平滑速率和水位阈值决策切换方向
3. 仅写 `_pending_action` flag，**不直接执行**切换
4. 在 `on_before_fit_step()` 被调用时（训练边界、Train GPU 空闲）才实际执行
    
**调度逻辑**：

```mermaid
flowchart TD
    A[监控循环] --> B{队列利用率?}
    B -->|> high_watermark\n队列过满| C{min_rollout_resources\n约束?}
    B -->|< low_watermark\n队列过空| D{min_train_resources\n约束?}
    B -->|水位正常| E[stable, 不切换]
    C -->|可切换| F[_pending_action = scale_train\n弹性资源 → Train]
    C -->|已达下限| E
    D -->|可切换| G[_pending_action = scale_rollout\n弹性资源 → Rollout]
    D -->|已达下限| E
    F --> H[等待训练边界触发]
    G --> H
    H --> I[on_before_fit_step\n冷却时间检查]
    I -->|cooldown 通过| J[_execute_action\n委托给 ElasticTrainer]
    I -->|cooldown 未过| E
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `high_watermark` | 0.8 | 队列超过此利用率 → 切换弹性资源为 Train |
| `low_watermark` | 0.3 | 队列低于此利用率 → 切换弹性资源为 Rollout |
| `cooldown_seconds` | 30.0 | 两次切换之间的最小间隔 |
| `min_rollout_resources` | 0 | 始终保持 Rollout 模式的最小弹性资源数 |
| `min_train_resources` | 0 | 始终保持 Train 模式的最小弹性资源数 |
| `ema_alpha` | 0.3 | 速率 EMA 平滑系数 |
| `confidence_threshold` | 0.6 | 执行切换所需的最低置信度 |

### 2.2 ElasticTrainer

`ElasticTrainer` 继承自 `FullyAsyncTrainer`，拥有完整角色切换序列的执行权。

**新增 API**：

| 方法 | 说明 |
|------|------|
| `switch_elastic_to_rollout(resource_id, param_version)` | 完整 Train→Rollout 序列 |
| `switch_elastic_to_train(resource_id, param_version)` | 完整 Rollout→Train 序列 |
| `add_elastic_actor(resource_id, world_ranks)` | 纳入新 Train rank，触发 DP rebuild |
| `remove_elastic_actor(resource_id)` | 排除 Train rank，触发 DP rebuild |
| `get_total_consumed_samples()` | 供 Coordinator 轮询消费速率 |

**DP Rebuild 触发时机**：切换发生在 `_fit_generate()` 收满数据后、GPU 计算启动前，此时 Train GPU **完全空闲**，重建成本为零。

### 2.3 ElasticActorWorker

`ElasticActorWorker` 继承自 `ActorRolloutRefWorker`，仅管理**训练引擎状态**。Rollout 服务器生命周期（wake_up / sleep / abort）完全由 `ElasticAgentLoopManager` 管理。

```mermaid
classDiagram
    class ElasticActorWorker {
        +ElasticState _elastic_state
        +init_model()
        +switch_to_train(new_world_ranks, param_version) bool
        +switch_to_rollout(param_version) bool
        +rebuild_dp_group(new_world_ranks)
        +get_elastic_state() dict
        -_patch_engine_to_elastic()
        -_offload_actor_to_cpu()
        -_load_actor_to_gpu()
        -_rebuild_dp_group(new_world_ranks)
    }
    class ElasticState {
        +str resource_id
        +ElasticMode current_mode
        +int param_version
        +list train_world_ranks
        +float last_switch_time
        +int total_switches
        +bool is_healthy
        +record_switch(new_mode)
    }
    class ElasticMode {
        <<enumeration>>
        TRAIN
        ROLLOUT
        SWITCHING
    }
    ElasticActorWorker --> ElasticState
    ElasticState --> ElasticMode
```

Engine 通过 `_patch_engine_to_elastic()` 动态注入弹性能力：在 `init_model()` 时将原始引擎的 `__class__` 替换为带 `ElasticMixin` 的子类（`ElasticMegatronMixin` 或 `ElasticFSDPMixin`），无需修改基础引擎代码。

### 2.4 ElasticRollouter

`ElasticRollouter` 继承自 `FullyAsyncRollouter`，通过替换 `ElasticAgentLoopManager` 获得弹性服务器管理能力。

**架构**：

```mermaid
graph TB
    ER["ElasticRollouter<br/>（Ray Actor，样本生产循环）"]
    EAM["ElasticAgentLoopManager<br/>（服务器池 + LB 管理）"]
    FR["Fixed Rollout Replicas<br/>始终活跃"]
    ELR["Elastic Rollout Replicas<br/>动态激活/休眠"]
    LB["ElasticGlobalRequestLoadBalancer<br/>（Ray Actor）"]
    W["AgentLoopWorker × N<br/>并发执行 AgentLoop"]

    ER --> EAM
    EAM --> FR
    EAM --> ELR
    EAM --> LB
    EAM --> W
```

**弹性副本预注册机制**：
- 启动时，所有弹性 `RolloutReplica` 在 `ElasticAgentLoopManager.create()` 中通过 `init_hybrid()` 初始化，随后立即 `sleep()`
- 注册进 `_registered_elastic_replicas` 但不加入 LB 池
- `add_elastic_replica()` 时 `wake_up()` + 加入 LB；`remove_elastic_replica()` 时触发 abort + `sleep()`

**ElasticGlobalRequestLoadBalancer** 在基础 LB 之上增加了 `_removed_servers` 集合，支持标记删除语义（mark-for-removal），使新请求不再路由到即将被移除的 server，同时存量请求继续正常完成。

---

## 3. DP 组动态重建

### 3.1 Megatron DP 重建流程

Megatron DP 每个 rank 持有**完整模型参数**（非分片），DP 组仅影响梯度 all-reduce。
        
```mermaid
sequenceDiagram
    participant ALL as 所有全局 Rank
    participant OLD as 老成员 rank
    participant NEW as 新成员 rank
    
    Note over ALL: rebuild_dp_group(new_world_ranks) 被所有 rank 调用
        
    ALL->>ALL: Step 1: capture_state_to_cpu()\n每个 rank 独立将本地模型/优化器快照到 CPU\n（无网络通信）
        
    ALL->>ALL: Step 2: _destroy_parallel_groups()\n销毁旧 TP/PP/DP 通信组

    Note over ALL: 集合操作窗口开始（所有 rank 必须对称参与）
    ALL->>ALL: Step 3a: dist.new_group(new_world_ranks)\n预建 sync_group（含排除的 rank 也必须调用）
    ALL->>ALL: Step 3b: _reinitialize_parallel_groups()\n含内部 dist.new_group 调用
    ALL->>ALL: Step 4: dist.barrier()
    Note over ALL: 集合操作窗口结束

    NEW->>NEW: Step 5: return（不在新组内的 rank 提前退出）

    OLD->>OLD: Step 6: _restore_state_from_cpu()\n从 CPU 快照恢复到 GPU

    OLD->>OLD: Step 7: _sync_params_to_new_members()\nsrc=new_world_ranks[0]\ndist.broadcast → 覆盖新成员的随机权重

    OLD->>OLD: Step 8: dist.barrier()
```

**Collective Barrier 对称性**（关键约束）：

`dist.new_group()` 和 `mpu.initialize_model_parallel()` 内部均包含集合操作，**所有全局 rank 必须参与相同数量的 `dist.new_group` 调用**，否则 NCCL 死锁。因此 Step 3 被提前到 `is_in_new_group` 判断之前执行，被排除的 rank 也会执行这两个集合操作后才 return。

### 3.2 FSDP2 DP 重建流程

FSDP2 参数以 `DTensor` 分片存储，重建时需先 `unshard` 再 offload。

```mermaid
sequenceDiagram
    participant ALL as 所有全局 Rank
    participant OLD as 老成员 rank
    participant NEW as 新成员 rank

    ALL->>ALL: Step 1: _save_state_to_cpu()\nunwrap FSDP，to_local() 获取本地 shard，拷贝到 CPU

    ALL->>ALL: Step 2: _destroy_parallel_groups()\n销毁旧 device_mesh

    ALL->>ALL: Step 3: dist.new_group(new_world_ranks)\n重建 DP 通信组

    ALL->>ALL: Step 4: dist.barrier()

    NEW->>NEW: Step 5: return（不在新组内的 rank 退出）

    OLD->>OLD: Step 6: _restore_state_from_cpu()\n重新包装 FSDP2（新 device_mesh）\n从 CPU 恢复参数

    OLD->>OLD: Step 7: _sync_params_to_new_members()\ndist.broadcast 覆盖新成员权重

    OLD->>OLD: Step 8: dist.barrier()
```

### 3.3 DP 扩缩容时参数分布（以 Megatron DP=2→4 为例）

```mermaid
flowchart TB
    subgraph BEFORE ["扩容前（DP=2，新成员持有随机权重）"]
        direction LR
        r0["rank0 GPU\n完整模型 W\n老成员 ✅"]
        r1["rank1 GPU\n完整模型 W\n老成员 ✅"]
        r2["rank2 GPU\n随机权重\n新成员 ❌"]
        r3["rank3 GPU\n随机权重\n新成员 ❌"]
    end

    subgraph AFTER ["broadcast 后（DP=4，全部同步）"]
        direction LR
        a0["rank0 GPU\nW  src ✅"]
        a1["rank1 GPU\nW  覆盖 ✅"]
        a2["rank2 GPU\nW  接收 ✅"]
        a3["rank3 GPU\nW  接收 ✅"]
    end

    r0 -->|"broadcast src=rank0\nnew dp_group"| a0
    r0 -->|broadcast| a1
    r0 -->|broadcast| a2
    r0 -->|broadcast| a3
```

| 阶段 | 说明 |
|------|------|
| CPU offload | 每个 rank 独立操作本地内存，**零网络通信** |
| 新成员 CPU 快照 | 仅用于恢复 tensor shape/dtype，实际值会被 broadcast 覆盖 |
| broadcast src | `new_world_ranks[0]`（全局 rank 0），在新建的 `sync_group` 内广播 |
| 缩容（DP=4→2） | 无新成员，无 broadcast，每个 rank 从自己 CPU 快照恢复，参数不变 |

---

## 4. AgentLoop 中 Server 删除的并发安全分析

### 4.1 Server 删除执行顺序

`remove_elastic_replica()` 严格按如下顺序执行：

```mermaid
sequenceDiagram
    participant MGR as ElasticAgentLoopManager
    participant LB as ElasticGlobalRequestLoadBalancer
    participant W as AgentLoopWorker × N
    participant ENG as vLLM / SGLang Engine
    participant FA as FullyAsyncLLMServerManager

    MGR->>LB: Step 1: remove_server(addr)\n标记 _removed_servers，新请求不路由到此
    MGR->>W: Step 2: notify_workers_server_removed(addr)\nawait 所有 worker ack\n_server_id_to_handle.pop(addr)
    Note over W: 此 await 保证 handle 在 abort 前已删除
    MGR->>ENG: Step 3: abort_all_requests()\npause_generation → stop_reason="aborted"
    Note over FA: FullyAsyncLLMServerManager while 循环检测到\nstop_reason="aborted"，触发重试
    FA->>LB: acquire_server(request_id)\nA 在 _removed_servers → 路由到健康 server B
    FA->>W: _server_id_to_handle.get(B) → handle B ✅
    FA->>ENG: 用新 server B 续推（prompt + 已生成 tokens）
    MGR->>LB: Step 4: cleanup_removed_server(addr)\n从 _inflight_requests 彻底删除
    MGR->>ENG: Step 5: replica.sleep()\n释放显存，GPU 归还 Training Engine
```

### 4.2 并发安全性分析

`generate()` 执行期间并发触发 `remove_server` 的各类 race 分析：

| 场景 | 是否安全 | 原因 |
|------|---------|------|
| `generate()` 已持有 server handle，同时 `_server_id_to_handle` 删除该 key | ✅ 安全 | `server` 是已取出的 Python 引用，dict 删除 key 不影响引用有效性 |
| `abort` 返回 `"aborted"`，重试时 `_server_id_to_handle[A]` 已被删除 | ✅ 安全 | Step 2 用 `await asyncio.gather` 阻塞到所有 worker ack，**保证 handle 先删再 abort** |
| sticky session 重试时 LB 仍返回被删 server A | ✅ 安全 | `acquire_server` 检查 `_removed_servers`，A 已在其中，重新路由到 B |
| `_release_server(A)` (fire-and-forget) 与 `cleanup_removed_server(A)` 并发 | ✅ 安全 | `ElasticGlobalRequestLoadBalancer.release_server` 对不存在的 key 静默返回，不抛异常 |
| 并发调用 `remove_elastic_replica` 两次（同时修改 `server_addresses` list） | ⚠️ 理论不安全 | Python list `pop` 非原子；但 `ElasticCoordinator` 通过 `_switch_lock` 保证串行调度，实践中不会并发 |

### 4.3 `partial_rollout` 重试机制

`FullyAsyncLLMServerManager.generate()` 对 AgentLoop 透明地完成断点续推：

```mermaid
flowchart TD
    A["generate(request_id, prompt_ids)"] --> B["while True"]
    B --> C["super().generate()\nprompt_ids + final_output.token_ids"]
    C --> D{stop_reason?}
    D -->|EOS / length| E["return final_output ✅"]
    D -->|aborted + partial_rollout=True| F["token_ids 已累积，继续重试"]
    F --> G["update max_new_tokens\n= original - len(accumulated)"]
    G --> C
    D -->|aborted + partial_rollout=False| E
```

**关键细节**：重试时拼接 `prompt_ids + final_output.token_ids`，新 server 从断点位置续推，上层 AgentLoop 感知不到中断。

---

## 5. 文件结构

```
verl/experimental/elastic_scheduling/
├── README.md                          # 本文档
├── main.py                            # ElasticSchedulingTaskRunner（主入口）
├── coordinator.py                     # ElasticCoordinator（Ray Actor，调度决策）
├── elastic_trainer.py                 # ElasticTrainer（完整切换序列执行）
├── elastic_rollouter.py               # ElasticRollouter（样本生产，弹性副本管理）
├── elastic_engine_workers.py          # ElasticActorWorker（训练引擎弹性切换）
├── elastic_checkpoint_engine.py       # ElasticCheckpointEngine（参数同步扩展）
├── agent_loop/
│   ├── __init__.py
│   └── elastic_agent_loop.py          # ElasticAgentLoopManager
│                                      # ElasticGlobalRequestLoadBalancer
│                                      # ElasticAgentLoopWorker
├── engine/
│   ├── __init__.py                    # get_elastic_engine_cls() 工厂函数
│   ├── fsdp/
│   │   └── elastic_transformer_impl.py  # ElasticFSDPMixin
│   │                                    # ElasticFSDPEngineWithLMHead
│   └── megatron/
│       └── elastic_transformer_impl.py  # ElasticMegatronMixin
│                                        # ElasticMegatronEngineWithLMHead
├── config/
│   ├── elastic_ppo_trainer.yaml          # 多节点配置
│   └── elastic_ppo_trainer_single_node.yaml  # 单节点开发配置
└── test/
    ├── test_fsdp_dp_rebuild_real_model.py    # FSDP2 DP 重建全量测试
    ├── test_megatron_dp_rebuild_real_model.py # Megatron DP 重建全量测试（R1–R7）
    ├── test_elastic_scheduling.py
    └── test_elastic_scheduling_ray.py
```

---

## 6. 配置说明

```yaml
elastic_scheduling:
  # 调度水位（基于 MessageQueue 利用率）
  high_watermark: 0.8         # 超过此值 → 弹性资源切换为 Train（减少 Rollout）
  low_watermark: 0.3          # 低于此值 → 弹性资源切换为 Rollout（增加生产）

  # 资源保护下限
  min_rollout_resources: 0    # 至少保留 N 个弹性 Rollout（防止队列彻底断供）
  min_train_resources: 0      # 至少保留 N 个弹性 Train（防止训练停止）

  # 切换节奏控制
  cooldown_seconds: 30.0      # 两次切换之间的最小冷却时间（防止抖动）
  check_interval: 2.0         # 监控循环轮询间隔（秒）
  confidence_threshold: 0.6   # 执行切换所需的最低置信度

  # 速率估计
  ema_alpha: 0.3              # EMA 平滑系数（越小越平滑，反应越慢）
```

---

## 7. 关键设计决策

### 7.1 为什么选择训练前边界触发切换

- **Train GPU 完全空闲**：`_fit_generate()` 期间 Train Workers 仅等待队列数据，GPU 零占用，切换成本为零
- **监控与执行解耦**：后台监控循环只写 `_pending_action` flag，实际切换在安全窗口由 `on_before_fit_step()` 触发，避免异步竞争
- **自然对齐**：`_fit_generate()` 等待越久 = 生产越不足，与 `scale_rollout` 决策方向天然一致

### 7.2 FSDP2 与 Megatron 重建策略对比

| 特性 | FSDP2 | Megatron |
|------|-------|----------|
| 参数存储 | DTensor 分片（各 rank 持有部分） | 完整副本（各 rank 持有全量） |
| CPU offload | 需 `to_local()` 提取本地 shard | 直接 `param.data.cpu()` |
| Barrier 对称性要求 | 同 | 同（`dist.new_group` 必须全局对称） |
| 重建后 broadcast | 有（新成员接收老成员参数） | 有（新成员接收 rank0 参数） |


### 7.3 rollout 弹性与非弹性分配
你再看下AgentLoopManager的实现，弹性的rollout资源也要收敛到AgentLoop中，弹性的server_id, 和非弹性的 server_id 发生了冲突，
同时必须要先初始化弹性资源，之后再初始化非弹性资源，因为在多机多卡场景下，需要优先给训练分配资源，这样资源亲和性比较高，然后剩余的资源，
再分配给独立的资源，同时server_id 需要唯一。

即使亲和的问题可以解决，无论是独立资源先创建server，还是弹性资源先创建server, 全局只有一个agent_loop_manager, server_id 不能重复。


### 7.4 弹性资源对于样本数量的动态获取
由于DP发生改变，训练获取的样本数量需要可以整除DP


### 7.5 指标记录
* 弹性切换次数
* 弹性切换延迟
* rollout 资源数 (包含独占资源)
* trainer 资源数

### 7.6 样本 Staleness 处理

弹性切换可能导致 Rollout 参数版本落后：
- 每个样本携带生成时的 `param_version`
- Trainer 计算 `stale_delta = current_version - sample_version`
- `stale_delta > staleness_threshold` 时丢弃该样本（与 `FullyAsyncTrainer` 保持一致）

---

## Q&A

### Q: DP 2→4 时，rank2/3 的 CPU 快照里是随机权重，`restore_from_cpu()` 之后它们的 GPU 参数正确吗？

**A**：不正确——`restore_from_cpu()` 只是把 CPU 快照（随机值）搬回 GPU。之后 Step 7 的 `_sync_params_to_new_members()` 用 `dist.broadcast(src=rank0)` 覆盖 rank2/3 的权重，使它们与 rank0 一致。rank2/3 的 CPU 快照仅用于重建 tensor 的 shape/dtype 结构，实际数值无关紧要。

### Q: 为什么不直接从 rank0 通过网络发给 rank2/3，跳过 CPU offload？

**A**：Megatron 的并行组（DP/TP/PP）是进程级全局单例，**必须先销毁旧组再重建新组**。销毁期间 NCCL 通信不可用，因此只能先把参数存到本地 CPU，等新组建立后再广播。

### Q: `remove_elastic_replica` 执行期间，正在 `generate()` 的请求会丢失吗？
 
**A**：不会。依赖三层保障：
1. LB 的 `_removed_servers` 标记使新请求不路由到被删 server
2. `notify_workers_server_removed()` 的 `await` 保证 handle 在 `abort` 前已从所有 worker 删除
3. `FullyAsyncLLMServerManager` 检测到 `stop_reason="aborted"` 后，自动将 `prompt + 已生成 tokens` 拼接发给新 server 续推，AgentLoop 无感知

