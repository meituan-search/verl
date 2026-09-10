# verl 对 Megatron 的使用

**核心文件**：[verl/workers/engine/megatron/transformer_impl.py](../../verl/workers/engine/megatron/transformer_impl.py)（~1240 行，MegatronEngine 本体）、[verl/workers/engine/base.py](../../verl/workers/engine/base.py)（engine 抽象与注册表）、[verl/workers/engine_workers.py](../../verl/workers/engine_workers.py)（worker 层）、[verl/workers/config/engine.py](../../verl/workers/config/engine.py)（Megatron 引擎配置）、[verl/utils/megatron_utils.py](../../verl/utils/megatron_utils.py)（模型构建/offload/训练 hooks）、[verl/utils/megatron/](../../verl/utils/megatron/)（optimizer / tensor_parallel / pipeline_parallel / sequence_parallel 工具）、[verl/utils/checkpoint/megatron_checkpoint_manager.py](../../verl/utils/checkpoint/megatron_checkpoint_manager.py)（~1300 行，checkpoint 管理）、[verl/workers/utils/losses.py](../../verl/workers/utils/losses.py)（loss 函数）

**核心问题**：verl 是一个 RL/RLHF 后训练框架。它需要 Megatron 的大模型并行训练能力——TP/PP/VPP/CP/EP 五维并行、DDP + DistributedOptimizer、1F1B pipeline schedule、dist checkpointing——但**完全不需要** Megatron-LM 的训练循环、数据管线和上千个 CLI 参数（那套体系见 [Megatron-LM 笔记 08](/Users/wangshulin/Desktop/Megatron-LM/study/notes/08-arguments-and-config-management.md)）。verl 的答案是：**只消费 `megatron.core`（mcore），完全绕过 `megatron.training` 层**；HF↔Megatron 的配置与权重双向转换交给独立的 Megatron-Bridge 库；Megatron 被封装成 EngineRegistry 里可插拔的一个 "engine" 后端，与 FSDP/torchtitan/VeOmni 同台竞争；所有后端共享同一套 loss 归一化协议，loss 函数对后端无感知。

---

## 一、全景：verl 用的是 Megatron 的哪一部分

### 1.1 与 Megatron-LM pretrain 用法的对照

Megatron-LM 自己用 Megatron 的方式是"全家桶"：`pretrain_gpt.py` → `parse_and_validate_args` → `initialize_megatron` → `setup_model_and_optimizer` → 训练循环。verl 只挑走了其中并行训练的骨架：

| Megatron 能力 | Megatron-LM 的入口 | verl 是否使用 | verl 的替代/入口 |
|---|---|---|---|
| 参数体系（argparse/Namespace） | `megatron/training/arguments.py` | **不用** | hydra yaml → verl dataclass → Megatron-Bridge（§三） |
| 并行状态（进程组） | `initialize_megatron` 内部 | **用** | `mpu.initialize_model_parallel` 直调（transformer_impl.py:156） |
| 模型构建 | `setup_model_and_optimizer` | **不用** | Megatron-Bridge `provider.provide_distributed_model`（§四） |
| Pipeline 调度 | `get_forward_backward_func` | **用** | 同一个 API，但喂的是 verl 的 forward_step（§五） |
| 优化器 | `get_megatron_optimizer` | **用** | 同一个 API，config 由 verl 映射构造（§六） |
| Loss 归约 | `loss_func` + pp schedule 内部平均 | **改造** | verl 的 loss × num_micro_batch 预乘对冲（§五） |
| 训练 hooks | `training.py` 注册 | **复刻** | `register_megatron_training_hooks`（megatron_utils.py:1541） |
| Dist checkpointing | `megatron/training/checkpointing` | **用**（mcore 版） | `megatron.core.dist_checkpointing` + verl 策略包装（§九） |
| 数据/词表/评估 | `megatron/training/*` | **不用** | verl 自己的 RL 数据流 |

一句话：**verl 把 mcore 当作一个"并行训练库"而非"训练框架"来用**。`megatron.training`（注意命名空间：这是 Megatron-LM 仓库的训练层，不是 mcore）里的 argparse 体系、`global_vars` 单例、pretrain 主循环，在 verl 中一处都没有出现。

### 1.2 分层图

```
┌────────────────────────────────────────────────────────────────────┐
│ ① 用户配置    hydra yaml（ppo_trainer.yaml，model_engine=megatron） │
│               → McoreActorConfig / McoreCriticConfig                │
├────────────────────────────────────────────────────────────────────┤
│ ② verl worker 层   TrainingWorker + ActorRolloutRefWorker           │
│               （Ray actor；actor/ref/critic/rollout 的编排）        │
├────────────────────────────────────────────────────────────────────┤
│ ③ verl engine 抽象  BaseEngine ← EngineRegistry("megatron")         │
│               MegatronEngine / WithLMHead / WithValueHead           │
├────────────────────────────────────────────────────────────────────┤
│ ④ Megatron-Bridge（外部库）                                         │
│   AutoBridge.from_hf_pretrained → MegatronProvider                 │
│   TransformerConfig 构造 / HF 权重加载 / HF 权重导出（双向）         │
├────────────────────────────────────────────────────────────────────┤
│ ⑤ megatron.core（mcore）                                            │
│   parallel_state / pipeline_parallel.schedules / DDP /             │
│   DistributedOptimizer / tensor_parallel / dist_checkpointing      │
└────────────────────────────────────────────────────────────────────┘
```

verl 与 Megatron 的全部接触面收敛在**五个协议点**上：

1. **配置协议**：verl dataclass → bridge provider overrides → `TransformerConfig`（§三）；
2. **模型协议**：bridge 构建并 DDP 包装好的 model chunks 列表（§四）；
3. **schedule 协议**：verl 的 `forward_step` 函数满足 mcore `forward_step_func` 的返回契约（§五）；
4. **优化器协议**：mcore `OptimizerConfig` + 训练 hooks 注册（§六）；
5. **导出协议**：bridge `export_hf_weights` 产出 `(hf_name, tensor)` 生成器，供权重同步与 checkpoint 复用（§八、§九）。

---

## 二、Engine 抽象：Megatron 只是后端之一

### 2.1 `BaseEngine`：后端无关的训练引擎接口

[BaseEngine](../../verl/workers/engine/base.py#L30)（base.py:30）定义了所有训练后端（megatron/fsdp/fsdp2/torchtitan/veomni/automodel）的统一接口：

| 方法 | 行号 | 职责 |
|---|---|---|
| `initialize()` | base.py:38 | 构建模型、优化器、lr scheduler、checkpoint manager |
| `train_batch(data, loss_function)` | base.py:113 | 模板方法：zero_grad → forward_backward → optimizer_step |
| `infer_batch(data, loss_function)` | base.py:134 | `torch.no_grad()` 下的 forward-only（log_prob 计算） |
| `forward_backward_batch(...)` | base.py:99 | 核心计算入口，**后端差异全部封装在这里** |
| `get_per_tensor_param()` | base.py:151 | 权重导出（HF 坐标），供 rollout 同步 |
| `save/load_checkpoint` | base.py:244 | checkpoint 契约 |
| `to(device, ...)` | base.py:231 | offload/设备迁移 |
| `train_mode()/eval_mode()` | base.py:58 | 上下文管理器，进入时自动从 CPU 加载需要的部分 |

### 2.2 `EngineRegistry`：字符串 → 引擎类

[EngineRegistry](../../verl/workers/engine/base.py#L330)（base.py:330）是三层嵌套字典注册表：`_engines[model_type][backend][device_key] -> engine class`。Megatron 的注册点只有两处（都在 [transformer_impl.py](../../verl/workers/engine/megatron/transformer_impl.py)）：

```python
@EngineRegistry.register(model_type="language_model", backend="megatron")   # :885
class MegatronEngineWithLMHead(MegatronEngine): ...

@EngineRegistry.register(model_type="value_model", backend="megatron")      # :1201
class MegatronEngineWithValueHead(MegatronEngineWithLMHead): ...
```

- `model_type` 区分任务头（LM head vs value head），由 worker 构造时传入；
- `backend` 即 hydra 配置里的 `strategy="megatron"`（McoreEngineConfig.strategy，engine.py:225）；
- 设备维度由 `get_device_name()/get_vendor()` 探测——NPU 上查 `"megatron"` 会命中 MindSpeed 子类（[mindspeed/transformer_impl.py:61](../../verl/workers/engine/mindspeed/transformer_impl.py#L61) 以 `device="npu"` 注册同名 backend），即 **MindSpeed 通过抢占设备键位"顶替"Megatron，worker 层零改动**。

消费点在 [TrainingWorker.\_\_init\_\_](../../verl/workers/engine_workers.py#L76)（engine_workers.py:135）：

```python
self.engine: BaseEngine = EngineRegistry.new(
    model_type=..., backend=self.engine_config.strategy, ...)
```

### 2.3 Worker 层的组合

[ActorRolloutRefWorker](../../verl/workers/engine_workers.py#L446)（engine_workers.py:446）是 PPO 的混合 worker，内部组合三个角色：

```
ActorRolloutRefWorker
 ├─ actor: TrainingWorker（MegatronEngineWithLMHead，forward_only=False → 带 DDP/优化器）
 ├─ ref:   TrainingWorker（同引擎类，但 engine_config.forward_only=True → 不建 DDP/优化器）
 └─ rollout: BaseRollout（vLLM/SGLang，与训练引擎同进程 colocate）
```

三条路径共用同一套 schedule（§五），区别只在标志位：

| 路径 | 入口（engine_workers.py） | 引擎行为 |
|---|---|---|
| actor 训练 | `update_actor` :702 → `actor.train_mini_batch` :241 | `train_batch`：forward_backward(forward_only=False) + optimizer_step |
| actor 旧 log_prob | `compute_log_prob` :694 → `actor.infer_batch` :391 | forward-only，`_lm_head_logits_processor` 算 log_probs |
| ref log_prob | `compute_ref_log_prob` :687 | 同上，走 ref 引擎 |
| critic 训练/推理 | 独立 TrainingWorker（model_type="value_model"） | `MegatronEngineWithValueHead.forward_step`（:1204）直接返回 values |

值得注意的细节：**ref 引擎与 actor 引擎在同一 Ray 进程里共享进程组**——`_init_device_mesh`（transformer_impl.py:139）开头就是 `if mpu.is_initialized(): return`，所以 actor/critic/ref 三个引擎只初始化一次 Megatron parallel_state（代价是三者并行度必须一致，代码里的 TODO 也承认了这点，:140）。

---

## 三、配置桥接：hydra → dataclass → Megatron-Bridge

### 3.1 配置链路

```
ppo_trainer.yaml（defaults 里 - model_engine: dp，可被覆盖为 megatron）
  → actor@actor_rollout_ref.actor: ${model_engine}_actor
    → trainer/config/actor/megatron_actor.yaml（_target_: ...McoreActorConfig）
      → 内嵌 engine（trainer/config/engine/megatron.yaml，_target_: McoreEngineConfig）
```

[McoreEngineConfig](../../verl/workers/config/engine.py#L150)（engine.py:150）就是 verl 侧的 Megatron 配置全集：

| 字段族 | 字段（engine.py:184-211） | 说明 |
|---|---|---|
| 并行度 | `tensor_model_parallel_size` / `expert_model_parallel_size` / `expert_tensor_parallel_size` / `pipeline_model_parallel_size` / `virtual_pipeline_model_parallel_size` / `context_parallel_size` | 五维并行，**DP size 仍是除法余数**，由 mpu 派生 |
| SP/CP | `sequence_parallel`（默认 True）、`dynamic_context_parallel`、`max_seqlen_per_dp_cp_rank` | TP=1 时 `__post_init__` 强制关 SP（:225-227） |
| 优化器/ckpt | `use_distributed_optimizer`、`use_dist_checkpointing`、`dist_checkpointing_path`、`dist_ckpt_optim_fully_reshardable` | |
| 逃生口 | `override_transformer_config` / `override_ddp_config` / `override_mcore_model_config`（dict） | **任意键值直透 Megatron config 的最终覆盖**，优先级最高 |
| bridge | `use_mbridge`、`vanilla_mbridge`（旧路径，已弃用警告）、`use_megatron_fsdp` | |

这套设计与 Megatron-LM 的"上千 CLI 参数"形成鲜明对比：verl 把 Megatron 的 243 字段 TransformerConfig **折叠成 ~30 个用户可见字段 + 一个任意覆盖 dict**。想调任何没暴露的旋钮（如 `recompute_method`），走 `override_transformer_config: {recompute_method: uniform}` 即可，不需要 verl 加字段。

### 3.2 `_build_tf_config`：从 HF config 到 TransformerConfig

[transformer_impl.py:168-288](../../verl/workers/engine/megatron/transformer_impl.py#L168)。**没有 argparse Namespace，没有 `core_transformer_config_from_args` 那套同名拷贝**——配置转换整体外包给 Megatron-Bridge：

```python
bridge = AutoBridge.from_hf_pretrained(model_config.local_path, ...)   # :210  读 HF config
provider = bridge.to_megatron_provider(load_weights=False)            # :214  造 provider
provider_overrides = {                                                # :220  verl 强制值
    "tensor_model_parallel_size": ..., "sequence_parallel": ...,
    "variable_seq_lengths": True,          # RL 序列长度天然可变
    "attention_backend": AttnBackend.flash,
    "moe_token_dispatcher_type": "alltoall",
    "moe_router_load_balancing_type": "none",   # 关掉 aux loss：RL 里伤性能
    "batch_p2p_comm": False,               # P2P 传 shape，不传 batch
    "overlap_p2p_comm": (vpp_size > 1),    # 仅 interleaved 时开
    ...
}
for key, value in override_transformer_config.items():                # :237  用户覆盖最后写入
    provider_overrides[key] = value
provider.apply_overrides_and_finalize(dtype=..., overrides=provider_overrides)  # :259
```

三层优先级清晰：**bridge 从 HF config 推导的默认值 < verl 强制值（RL 场景必需）< 用户 `override_transformer_config`**。

几个值得注意的强制覆盖：

- **value model**：强制 `tie_word_embeddings=False`（transformer_impl.py:181-186），value head 不能和词表 embedding 共享权重——必须在 bridge finalize 之前改；
- **dynamic CP 的"拆东墙补西墙"**（:187-194）：`max_seqlen_per_dp_cp_rank` 透传，但把 `dynamic_context_parallel` 改成 False、`context_parallel_size` 改写成 DP world size——绕开 Megatron-LM 上游的耦合 bug（注释里贴了上游代码链接）；
- **MoE router replay**（:248-252）：按 provider 属性版本差异写 `moe_enable_routing_replay` 或 `enable_routing_replay`。

### 3.3 优化器与 DDP 配置的映射

[mcore OptimizerConfig](../../verl/utils/megatron/optimizer.py#L25) 由 [init_megatron_optim_config](../../verl/utils/megatron/optimizer.py#L25)（optimizer.py:25-98）从 verl 的 `McoreOptimizerConfig` 逐字段翻译：

- 通用超参：`optimizer/lr/min_lr/clip_grad/weight_decay/use_distributed_optimizer`（:31-38）；
- **fp16 分支**自动开 `use_precision_aware_optimizer=True` + `store_param_remainders=False`（:40-50）——fp16 必须有 fp32 master weight，这是 Megatron 的硬约束；
- **bf16 分支**默认保持 fp32 动量，可选 precision-aware（`main_grads_dtype`/`exp_avg_dtype` 可降 fp8，:68-76）；
- `override_optimizer_config` dict 同样是最终覆盖（:90-93）。

DDP config 有一个联动细节（[_resolve_override_ddp_config](../../verl/workers/engine/megatron/transformer_impl.py#L290)，transformer_impl.py:290-310）：precision-aware optimizer 且 `main_grads_dtype < fp32` 时，自动注入 `grad_reduce_in_fp32=False`，保持 DDP grad bucket 与优化器 grad buffer dtype 一致。

---

## 四、模型构建与权重加载

### 4.1 `make_megatron_module`：组装的核心

[make_megatron_module](../../verl/utils/megatron_utils.py#L220)（megatron_utils.py:220-368）是模型构建的枢纽，Megatron-Bridge 路径下分三步：

1. **注册 pre-wrap hook**（:269-298）：PEFT（LoRA）、value model 改造（`make_value_model`）、freeze MoE router——全部在 DDP 包装**之前**做。注释解释了原因（:260-265）：frozen 参数若进了 DDP grad bucket 会引发 KeyError，分布式优化器也只应跟踪 adapter 参数；
2. **`create_ddp_config` + `provider.provide_distributed_model(wrap_with_ddp=..., ddp_config=...)`**（:301-316）：由 bridge 产出 mcore 模型。`wrap_with_ddp = not engine_config.forward_only`（transformer_impl.py:316-319）——ref 引擎连 DDP 都不包；
3. **权重加载**（transformer_impl.py:343-356）：dist ckpt 走 `load_mcore_dist_weights`，否则 `bridge.load_hf_weights(module, local_path)`——HF→Megatron 的名字映射与 TP/PP/EP reshard **全部在 bridge 内部完成**，verl 不写切分逻辑。

### 4.2 `get_model`：移植 Megatron-LM 的 pretrain 构建逻辑

非 bridge 路径（vanilla mbridge / 离线工具）走 [get_model](../../verl/utils/megatron_utils.py#L58)（megatron_utils.py:58-175），这是 Megatron-LM `setup_model_and_optimizer` 中模型部分的移植：VPP 时循环构造多个 model chunk（:77-86）、`Float16Module` 包裹（:145-146）、mcore `DistributedDataParallel` 包裹 + `broadcast_params`（:148-174）。配套的 [model_initializer.py](../../verl/models/mcore/model_initializer.py) 负责选择 layer spec（dense 用 `get_gpt_decoder_block_spec`、DeepSeekV3 用 `get_gpt_mtp_block_spec`，:102/:191），[config_converter.py](../../verl/models/mcore/config_converter.py) 按 HF `model_type` 分派到各家族的转换函数（注册于 [registry.py](../../verl/models/mcore/registry.py)）。这套直构路径现在主要服务于离线工具（model merger）；在线训练默认走 bridge。

### 4.3 `patch.py`：版本兼容的"负空间"

mcore 迭代很快，verl 用 [patch.py](../../verl/models/mcore/patch.py) 维护跨版本可用性，手法是**按版本条件打补丁、优先信任新版上游**：

- `apply_patch()`（:20）：修 mcore 0.12 MLA 的 `get_query_key_value_tensors` 在 `packed_seq_params` 非 None 时出错的 bug（整函数重写，:39-229）；但 `mcore_ge_013` 就不再覆盖（:356-357）；
- `apply_patch_megatron_v012_with_torch_v28_v29`（:393）：修 torch 2.8/2.9 + mcore 0.12.1 的 dist-ckpt 序列化崩溃，在包 import 时即生效；
- `apply_patch_megatron_recomputation_backward`（:514）：MoE + activation checkpointing 时重写 `CheckpointFunction.backward` 显式释放驻留显存的输入/梯度（引用 Megatron-LM PR#3267）。

### 4.4 forward 三件套：把 logits 的消费权从 Megatron 拿回 verl

Megatron-LM 的 `forward_step` 会在模型内部算 loss（`loss_func` 协议）；verl 需要的是**原始 logits 消费权**（要算 log_prob/entropy/value/custom loss）。[model_forward.py](../../verl/models/mcore/model_forward.py) 的 `gptmodel_forward_model_engine`（:264）为此重写了 GPTModel 的调用序列：THD 打包（`preprocess_thd_engine`）→ `model(input_ids, packed_seq_params=...)` → 在 post_process stage 调用**外部传入的 `logits_processor`**。于是 loss 计算留在 verl（losses.py），Megatron 只负责张量并行地跑 forward。另有 `model_forward_fused.py`（monkey-patch `GPTModel.forward` 配合 fused linear-cross-entropy kernel，:53-61）和 `model_forward_1f1b_overlap.py`（1F1B overlap 调度定制）两个变体。

---

## 五、训练循环：与 pipeline schedule 的对接

这是 verl 使用 Megatron 最精巧的部分。[forward_backward_batch](../../verl/workers/engine/megatron/transformer_impl.py#L671)（transformer_impl.py:671-808）把一个 DP-rank 本地的 mini batch 喂进 mcore 的 1F1B schedule。

### 5.1 前置：全局 token 统计与 micro batch 切分

```python
batch_num_tokens = data["loss_mask"].sum()                            # :676
torch.distributed.all_reduce(batch_num_tokens, group=dp_group)        # :677  DP 组全局 token 数
tu.assign_non_tensor(data, batch_num_tokens=..., dp_size=...)         # :680  塞进 TensorDict 供 loss 用

micro_batches, indices = prepare_micro_batches(                       # :709  切 micro batch
    data, dp_group, num_batches_divided_by, same_micro_num_in_dp=True)
```

两个要点：

- **loss 归一化的分母在 schedule 之前就算好**：所有 micro batch 的 loss 都除以全局 `batch_num_tokens`（跨 DP rank 的 all-reduce），保证梯度与"全局 batch 的 token 均值"一致，与 FSDP 后端数值等价；
- **micro batch 切分两条路**（[prepare_micro_batches](../../verl/workers/engine/utils.py#L57)）：静态按 `micro_batch_size_per_gpu` 等分；`use_dynamic_bsz=True` 时按 token 预算（`max_token_len_per_gpu`）贪心切分 + DP 组内对齐 micro batch 数量 + seqlen 负载均衡（[seqlen_balancing.py](../../verl/utils/seqlen_balancing.py)）。返回的 `indices` 用于最后把乱序结果还原（`restore_dynamic_batch`）。VPP>1 时还必须整除 `microbatch_group_size_per_vp_stage`（:717-721，interleaved 调度的硬约束）。

### 5.2 调用 schedule：三个技巧

```python
forward_backward_func = get_forward_backward_func()        # :731  mcore 原生选择器
batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))  # :757
losses_reduced = forward_backward_func(                    # :761
    forward_step_func=forward_step,                        #   verl 的闭包（见 5.3）
    data_iterator=batch_generator,
    model=self.module,                                     #   VPP 时是 model chunks 列表
    num_microbatches=n_micro_batch,
    seq_length=1,        # the communication shape is obtained via p2p comm
    micro_batch_size=1,  # the communication shape is obtained via p2p comm
    forward_only=forward_only)
```

**技巧一：`seq_length=1, micro_batch_size=1`**。这两个参数在 Megatron-LM 里用于预计算 PP stage 间激活通信的 shape；verl 走 THD（remove padding 打包变长序列）路径，真实 shape 通过 P2P 通信（`batch_p2p_comm=False`）动态传递，所以填 1 即可。这是 verl 能在 Megatron 上跑**变长 RL 序列**的关键。

**技巧二：不选调度器**。verl 从不指定 1F1B 还是 interleaved——`get_forward_backward_func()` 内部按 `virtual_pipeline_model_parallel_size` 决定；verl 只通过 `_init_device_mesh` 传入的 VPP 配置间接控制。唯一要配合的是 [make_batch_generator](../../verl/utils/megatron/pipeline_parallel.py#L49)（pipeline_parallel.py:49-71）：VPP>1 时把 micro batch 列表复制成 `vpp_size` 份迭代器（每个 model chunk 一份）。注意 [pipeline_parallel.py](../../verl/utils/megatron/pipeline_parallel.py) 全文只有 71 行——**verl 不 patch 任何 Megatron schedule 函数**，适配全靠参数技巧和协议遵守。

**技巧三：只有最后一个 PP stage 收集输出**（:801-802）：`mpu.is_pipeline_last_stage(ignore_virtual=True)` 的 rank 调 `postprocess_batch_func` 把各 micro batch 的输出拼回 jagged tensor 并按 `indices` 还原动态切分顺序，其它 rank 返回空 dict。

### 5.3 `forward_step` 与 loss 的交接协议

verl 的 [forward_step](../../verl/workers/engine/megatron/transformer_impl.py#L957)（MegatronEngineWithLMHead，:957-1110）每个 micro batch 被 schedule 调一次：

1. `next(batch_iter)` 取 micro batch（dynamic CP 时先 `dynamic_cp_split_batch`，:962-971）；
2. 调 `get_mcore_engine_forward_fn(hf_config)`（§4.4 的 forward 三件套）跑 THD forward；
3. 在 post_process stage 经 [_lm_head_logits_processor](../../verl/workers/engine/megatron/transformer_impl.py#L905)（:905-955）算 log_probs/entropy——这里调 [verl/utils/megatron/tensor_parallel.py](../../verl/utils/megatron/tensor_parallel.py) 的 `vocab_parallel_log_probs_from_logits`（:201，包装 mcore 的 `vocab_parallel_cross_entropy`）和 `vocab_parallel_entropy`（:109，自定义 autograd Function，在 TP 切分的词表维度上 all-reduce 求熵，backward 手工推导且原位复用 buffer）。**TP 分片 logits 从不物化全量词表**，这是 Megatron 后端相对 FSDP 的显存优势之一；
4. 返回 `(output, partial(postprocess_micro_batch_func, data=batch, ...))`——第二个元素就是 mcore `forward_step_func` 契约中的 `loss_func`。

[postprocess_micro_batch_func](../../verl/workers/engine/megatron/transformer_impl.py#L1112)（:1112-1198）是协议的另一半，做三件事：

**① loss 缩放对冲**（:1123-1126）：

```python
loss, metrics = loss_function(model_output, data, dp_group=...)
# scale loss by num_micro_batch because megatron will scale loss
# by n_micro_batch inside pp schedule
scaled_loss = loss * data["num_micro_batch"]
```

mcore 的 1F1B schedule 内部会把每个 micro batch 的 loss 除以 `num_microbatches`（梯度累积因子），verl 预先乘回去——**于是跨 micro batch 的梯度累加恰好等于 loss 本身的梯度**，归一化完全由 verl 的全局 token 数控制，Megatron 原生的 loss 平均被干净地绕开。这条约定写进了 [losses.py](../../verl/workers/utils/losses.py#L28) `sft_loss` 的注释和 [agg_loss](../../verl/trainer/ppo/core_algos.py#L1138)（core_algos.py:1138）的 docstring："FSDP: 直接 backward；Megatron: 需再乘 num_microbatches"——而乘法封装在引擎里，loss 函数本身对后端无感知。

**② forward-only 时返回 dummy loss**（:1127-1130）：`loss_function=None`（纯 log_prob 计算）时返回 `torch.tensor(1.0)`，真实结果已在 `model_output["log_probs"]` 里。schedule 需要一个可 backward 的标量，哪怕是常数。

**③ per-token-loss 三元组**（:1149-1195）：`calculate_per_token_loss=True`（Megatron-Bridge 在 CP>1 时自动开启）时返回 `(loss_sum, num_tokens, output)` 三段式——mcore 的 `finalize_model_grads` 会 all-reduce token 数并除到梯度上。这段代码的注释堪称 verl 最长的一份"防坑记录"：

- 三元组是**必须**的：MoE router 的 aux/z loss 被 mcore 预乘了 num_tokens，靠这个除法对冲，否则 CP>1 时 grad_norm 爆炸 ~1e4；
- 禁用 `seq-mean-token-mean` 聚合模式（其 per-sequence 1/n_s 在 CP 分片下与全局归一化冲突）；
- 强制 THD（`use_remove_padding=True`）：BSHD 下 router 会按含 padding 的 B*S 归一化 aux loss；
- 本地 token 数要除以 cp_size（真实 token 在 CP rank 间是复制的，不除会多数 cp_size 倍）。

### 5.4 一次 actor update 的完整链路

```
trainer（v1: trainer_base.py / v0: ray_trainer.py）
    ▼  update_actor(batch)  ── dispatch_mesh="actor"，按 DP 切到各 rank
TrainingWorker.update_actor（engine_workers.py:702）
    ▼  train_mini_batch（:241）── 按 mini_batch_size/epochs 迭代
BaseEngine.train_batch（engine/base.py:113）
    ├─ optimizer_zero_grad()          zero_grad + 每个 chunk 的 zero_grad_buffer()
    ├─ forward_backward_batch()       §5.1-5.3 的全部内容
    └─ optimizer_step()               optimizer.step() → (update_successful, grad_norm, num_zeros)
    ▼
postprocess_output（engine_workers.py:180）── DP 组 all-reduce loss/metrics，算 MFU
```

grad 的归约全程由 Megatron 接管：`finalize_model_grads`（通过 §六的 hooks 注册进 model config）负责 grad all-reduce；fp16 loss scaling 由优化器的 `grad_scale_func` 处理；verl 不做任何手动梯度操作。

---

## 六、优化器与训练 hooks

### 6.1 优化器：mcore 原生，参数分组不碰

[get_megatron_optimizer](../../verl/utils/megatron/optimizer.py#L101)（optimizer.py:101-110）就是 `megatron.core.optimizer.get_megatron_optimizer` 的直接转发。**参数分组、weight decay 分组、DoO 的状态分片全部由 mcore 内部决定**，verl 不构造 param groups。LR 调度用 [get_megatron_optimizer_param_scheduler](../../verl/utils/megatron/optimizer.py#L112)（包装 mcore 的 `OptimizerParamScheduler`，含 warmup ratio 与 WSD decay）。

clip_grad 同样交给 Megatron（`OptimizerConfig.clip_grad`）；`optimizer.step()` 返回三元组 `(update_successful, grad_norm, num_zeros_in_grad)`，verl 只把 grad_norm 写进 metrics（transformer_impl.py:529-548），失败直接 raise。`optimizer_zero_grad`（:519-527）除常规 zero_grad 外还逐 chunk 调 `zero_grad_buffer()`——DistributedOptimizer 的 grad buffer 不清零会累积。

### 6.2 `register_megatron_training_hooks`：复刻 Megatron-LM 的接线

Megatron-LM 在 `training.py` 里给 model config 挂的一批 hooks，verl 在 [register_megatron_training_hooks](../../verl/utils/megatron_utils.py#L1541)（megatron_utils.py:1541-1584）复刻：

```python
config = get_model_config(model)
config.grad_scale_func = optimizer.scale_loss                  # fp16 loss scaling
config.finalize_model_grads_func = finalize_model_grads        # grad all-reduce + per-token 除法
if overlap_grad_reduce:
    config.no_sync_func = ... ; config.grad_sync_func = ...    # DDP grad reduce 与计算重叠
if overlap_param_gather:
    config.param_sync_func = ...                               # DoO param all-gather 重叠
```

这些钩子是 mcore schedule 在正确时机回调的协议——verl 逐字复刻 Megatron-LM 的接法，而不是自己发明。这是"用库而非用框架"策略的边界：**凡是 mcore 设计为必须由调用方接线的部分，verl 都接了；凡是 Megatron-LM 训练循环私有的部分，verl 都不要**。

---

## 七、显存 offload

RL 训练的每个阶段（generate → old log_prob → ref log_prob → critic → actor update）用不同的模型组合，colocate 模式下其余模型必须让出显存。Megatron 侧的 offload 在 [megatron_utils.py](../../verl/utils/megatron_utils.py)：

- [offload_megatron_model_to_cpu](../../verl/utils/megatron_utils.py#L525)（:525-611）：把每个 DDP chunk 的 `buffer.param_data` 拷进**常驻 pinned cpu buffer**（一次性分配，避免反复 pin 造成 2× host 内存峰值，:540-573），然后 GPU storage `resize_(0)`；LoRA 冻结参数单独搬；TE FP8 workspace 显式清（:604-608）。[load_megatron_model_to_gpu](../../verl/utils/megatron_utils.py#L615) 反向恢复；
- [offload_megatron_optimizer](../../verl/utils/megatron_utils.py#L745)（:745-791）：处理 `ChainedOptimizer`、fp32 master weights（`shard_fp32_from_float16_groups`）、`exp_avg/exp_avg_sq` 状态、TE `_dummy_wgrads` 和 `get_global_memory_buffer()`；
- 触发时机由 `train_mode()/eval_mode()` 上下文自动管理（BaseEngineCtx，base.py:291）：进入时按需加载 param/grad/optimizer，退出时按配置卸回。

---

## 八、权重同步：Megatron → rollout 引擎

colocate 模式下每个训练步结束后要把新权重推给 vLLM/SGLang。入口 [ActorRolloutRefWorker.update_weights](../../verl/workers/engine_workers.py#L720)（engine_workers.py:720-804），Megatron 侧的导出在 [get_per_tensor_param](../../verl/workers/engine/megatron/transformer_impl.py#L810)（transformer_impl.py:810-839）：

```python
load_megatron_model_to_gpu(self.module, load_grad=False)   # :817  offload 的参数搬回
...
self.bridge.export_hf_weights(self.module)                 # :823  (hf_name, tensor) 生成器
```

**关键设计：同步一律以 HF 全量坐标为中间格式**。Megatron 的 TP/PP/EP 分片合并、fused QKV/gate_up 的拆分（按 query group / gate-up 交错布局），全部由 bridge 的 `export_hf_weights` 内部完成（对 MP rank 做 all-gather）。训练侧无论什么并行度，产出的都是推理引擎直接认识的 HF 命名权重。仓库里保留的等价参考实现是 [saver.py](../../verl/models/mcore/saver.py) 的 `merge_megatron_ckpt_gptmodel`（:83）：`_broadcast_tp_shard_tensor_qkv`（:271，QKV 按 query group 交错拆分）、`_broadcast_tp_shard_tensor_gate_up`（:221）——想搞懂 Megatron→HF 的切分布局，读这两个函数。

接收端（与 Megatron 无关，但决定导出格式）：

- **vLLM**：ZMQ + CUDA IPC 分桶传输，最终 `model.load_weights(param_updates)`，MoE 权重 loader 有专项 patch；
- **SGLang**：按字节分桶，走 SGLang 官方 `update_weights_from_tensor` HTTP/Ray 接口；
- **分离部署**（trainer 与 rollout 不同节点）：`CheckpointEngineRegistry`（engine_workers.py:680）选择 nccl/nixl 等后端，actor rank0 打包 bucket 经 `ray.util.collective` 广播。**delta_sharded 增量同步目前只支持 FSDP/VeOmni**（依赖 `get_per_tensor_param_shard`，MegatronEngine 未实现，base.py:161 抛 NotImplementedError）；
- **LoRA 不 merge**：`base_sync_done` 后走 adapter-only 两段同步（只导出 adapter 权重，:812-821）；
- **量化**：ModelOpt QAT 在导出时在线量化成 NVFP4（[qat_weight_exporter.py](../../verl/utils/modelopt/qat_weight_exporter.py)）；普通 FP8 则以 bf16 全量同步、**在推理端转换**——旧 sharding manager 的 FP8 逻辑随其弃用而移到了 rollout 侧。

---

## 九、Checkpoint：mcore dist checkpointing 之上的 v2 布局

[MegatronCheckpointManager](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L115)（~1300 行）建立在 `megatron.core.dist_checkpointing` 上：

**保存**（[save_checkpoint](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L1169)，:1169-1299）采用 v2 目录布局：

```
<ckpt_dir>/global_step_<n>/
 ├─ ckpt_contents.json          # manifest（rank0 最后原子写）
 ├─ model/dist_ckpt/            # Megatron 分片权重（每 VPP chunk 一棵，model / model{vpp_rank}）
 ├─ model/huggingface/          # bridge 导出的 HF 权重
 ├─ optimizer/dist_ckpt/        # optimizer + lr_scheduler（布局依赖 model 布局）
 ├─ extra/dist_ckpt/            # rng_state
 └─ transformer_config.json
```

底层经 [save_dist_checkpointing](../../verl/utils/megatron/dist_checkpointing.py#L29)（dist_checkpointing.py:29-53）：`torch_dist` 策略外包一层 `FullyParallelSaveStrategyWrapper(..., dp_cp_group)`——**DP 维度全并行写**，避免 rank0 瓶颈。异步保存在 mcore ≥0.14 走 `AsyncCallsQueue`，finalize 挂在最后一个请求上。RNG 状态保存（python/numpy/torch/TP rng_tracker/设备 rng，:268-305）覆盖了 Megatron 的 model-parallel RNG 语义。

**恢复**（[load_checkpoint](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L856)，:856-985）：读 manifest 分三段加载（model / optimizer+lr / rng），load 端同样用 `FullyParallelLoadStrategyWrapper`，并为 `torch.load` 注册 AdamW/FusedAdam safe globals。**resume 自 HF**（`load_contents` 含 `hf_model`）直接 `bridge.load_hf_weights`，与冷启动同一条路。

**离线转换**：[megatron_model_merger](../../verl/model_merger/megatron_model_merger.py) 把 dist ckpt 转回 HF 格式。里面有个实用技巧（:144-173）：单进程以 `pp=world_size` 起一组假并行，**用 PP 维度读分片**——因为 mcore dist ckpt 的分片布局天然支持任意 TP/PP 组合重切（`load_dist_checkpointing` 内部 reshard），用 PP=world_size 读就把各 rank 的层错开还原成全局层号。

---

## 十、完整生命周期总览

以 PPO colocate + Megatron actor 为例，一个训练步内 Megatron 相关的全部动作：

```
① 初始化（一次性）
TrainingWorker.__init__（engine_workers.py:76）
 └─ initialize_global_process_group_ray()          distributed.py:82   Ray 注入 RANK/WORLD_SIZE
 └─ EngineRegistry.new("language_model", "megatron")
     MegatronEngine.__init__（transformer_impl.py:79）
     ├─ mpu.initialize_model_parallel(TP,PP,VPP,CP,EP,ETP)   :156   全部并行进程组
     ├─ set_random_seed() → model_parallel_cuda_manual_seed  :95    MP 安全的 RNG
     └─ （reset() 时）initialize()
         ├─ _build_tf_config: AutoBridge.from_hf_pretrained  :168   HF→TransformerConfig
         │    + provider_overrides（verl 强制值 + 用户 override）
         ├─ make_megatron_module: pre-wrap hook(PEFT) → DDP  megatron_utils.py:220
         │    → bridge.load_hf_weights                        :347
         ├─ init_megatron_optim_config → get_megatron_optimizer  optimizer.py:25/101
         ├─ register_megatron_training_hooks                   megatron_utils.py:1541
         │    （grad_scale_func / finalize_model_grads / overlap hooks）
         └─ offload to CPU（按配置）

② 每步：generate（rollout，非 Megatron）→ old/ref log_prob → critic → actor update
engine.train_batch(data, ppo_loss)（engine/base.py:113）
 ├─ optimizer_zero_grad + zero_grad_buffer
 ├─ forward_backward_batch（transformer_impl.py:671）
 │    ├─ DP 组 all-reduce batch_num_tokens                  :677
 │    ├─ prepare_micro_batches（dynamic bsz / 静态）          :709
 │    ├─ make_batch_generator（VPP 复制迭代器）               :757
 │    └─ get_forward_backward_func()(                        :761   mcore 1F1B/interleaved
 │         forward_step=verl 闭包, seq_length=1, micro_batch_size=1)
 │         每个 micro batch:
 │           forward_step(:957) → THD forward → _lm_head_logits_processor(:905)
 │             → vocab_parallel_log_probs/entropy（TP 分片词表）
 │           postprocess_micro_batch_func(:1112)
 │             → ppo_loss(...) × num_micro_batch（对冲 schedule 内部平均）
 │             → (scaled_loss, output) 或 (loss_sum, num_tokens, output)@CP>1
 │         schedule 内部: P2P 传激活、backward、finalize_model_grads（grad all-reduce）
 ├─ optimizer_step → DistributedOptimizer（clip + fp32 master + DoO 状态更新）:529
 └─ postprocess_batch_func（最后 PP stage 聚合 + restore_dynamic_batch）:801

③ 每步末：权重同步
ActorRolloutRefWorker.update_weights（engine_workers.py:720）
 └─ engine.get_per_tensor_param → bridge.export_hf_weights   transformer_impl.py:810
 └─ rollout.update_weights（vLLM IPC / SGLang API / checkpoint_engine）

④ 定期：save_checkpoint
MegatronEngine.save_checkpoint → MegatronCheckpointManager     :1169
 └─ mcore dist_checkpointing + FullyParallelSaveStrategyWrapper(DP,CP 组)
```

---

## 十一、verl 消费的 Megatron API 清单

| mcore 模块 | verl 用到的符号 | 用途 |
|---|---|---|
| `parallel_state` | `initialize_model_parallel`、`is_initialized`、`get_data_parallel_rank/world_size/group`、`get_model_parallel_group`、`is_pipeline_last_stage`、`get_virtual_pipeline_model_parallel_world_size`、`get_global_memory_buffer` | 进程组初始化与并行身份查询 |
| `pipeline_parallel` | `get_forward_backward_func` | 1F1B / interleaved schedule（唯一入口，不 import 具体调度函数） |
| `distributed` | `DistributedDataParallel`、`DistributedDataParallelConfig`、`finalize_model_grads` | DDP 包装、grad 归约 |
| `optimizer` | `OptimizerConfig`、`get_megatron_optimizer`、`ChainedOptimizer`、`OptimizerParamScheduler` | 优化器全家桶 |
| `tensor_parallel` | `vocab_parallel_cross_entropy`、`gather/scatter_from_sequence_parallel_region`、`model_parallel_cuda_manual_seed` | TP 分片下的 log_prob/熵、SP 数据搬运、MP RNG |
| `transformer` | `TransformerConfig`/`MLATransformerConfig`、`GPTModel`、`Float16Module`、`AttnBackend`、`get_gpt_decoder_block_spec` 等 spec 工厂 | 模型本体 |
| `dist_checkpointing` | `save/load`、`ShardedObject`、`AsyncCallsQueue` | checkpoint |
| `utils` | `get_model_config`、`get_attr_wrapped_model` | hooks 接线、模型 unwrap |
| `package_info` | `__version__` | 版本 gating（patch 老版本 bug） |
| **Megatron-Bridge**（外部库） | `AutoBridge.from_hf_pretrained`、`to_megatron_provider`、`provide_distributed_model`、`load_hf_weights`/`export_hf_weights`、`create_ddp_config`、`peft.*` | HF↔Megatron 的配置/权重双向转换、模型构建、PEFT |

注意清单里**没有任何 `megatron.training` 命名空间的符号**——这是"只用 mcore"边界的直接证据。

---

## 十二、设计要点小结

| 设计 | 机制 | 收益 |
|---|---|---|
| 用库不用框架 | 只消费 `megatron.core` + Megatron-Bridge，绕过 `megatron.training` 的参数体系与训练循环 | verl 的 RL 数据流/loss 完全自主，Megatron 只提供并行训练原语 |
| 可插拔后端 | `EngineRegistry(model_type, backend, device)` 注册表；`strategy="megatron"` 只是配置里的一个字符串 | FSDP/torchtitan/VeOmni/MindSpeed 同接口替换；MindSpeed 靠设备键位"顶替"Megatron，worker 零改动 |
| 配置折叠 | ~30 个用户字段 + `override_*_config` 任意覆盖 dict；三层优先级（bridge 默认 < verl 强制 < 用户覆盖） | 不必为 mcore 的上千参数建镜像；任何新旋钮零改动可达 |
| 转换外包 | HF↔Megatron 的 config/权重双向转换全部交给 Megatron-Bridge | TP/PP/EP reshard、fused QKV 拆合等最易错的逻辑不在 verl 维护 |
| 参数技巧优于 monkey-patch | `seq_length=1`（P2P 传 shape）、`batch_p2p_comm=False`、`variable_seq_lengths=True`、VPP 迭代器复制；pipeline_parallel.py 仅 71 行 | 变长 RL 序列跑在原生 1F1B 上，上游升级几乎不受影响 |
| loss 协议对冲 | loss × `num_micro_batch` 预乘抵消 schedule 内部平均；`agg_loss` 按 `batch_num_tokens`/`dp_size` 归一 | 同一份 loss 函数对 FSDP/Megatron 数值等价，loss 层无后端分支 |
| 交接而非重写 | `finalize_model_grads`/`grad_scale_func` 等钩子逐字复刻 Megatron-LM 接法；版本 patch 按"新版上游优先"原则收回 | 优化器/梯度流的正确性站在 Megatron 已验证的行为上 |
| HF 坐标作通用语 | 权重导出（rollout 同步、HF checkpoint、merger）统一走 `export_hf_weights` 全量 HF 格式 | 训练并行度与推理引擎解耦；任何后端产出的权重长得一样 |
| 防坑注释即文档 | per-token-loss 三元组、SP 依赖 TP、PEFT 先于 DDP 等约束以长注释形式固化在代码现场 | 下一个改这些代码的人不必重新踩坑 |

一句话总结：**verl 把 Megatron 当作一个"并行训练库"而非"训练框架"来使用——mcore 提供进程组、schedule、DDP/DoO、dist checkpointing 四类原语，Megatron-Bridge 提供双向转换，其余一切（数据、loss、训练循环、权重同步）都由 verl 自己的 engine 抽象接管，并通过"loss 缩放对冲 + forward_step 协议 + HF 坐标导出"三个协议点把两边干净地缝合在一起。**
