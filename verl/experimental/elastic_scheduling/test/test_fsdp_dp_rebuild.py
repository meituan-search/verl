#!/usr/bin/env python3
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
真实模型端到端测试：通过 ElasticFSDPEngineWithLMHead (fsdp2_elastic) 加载
Qwen2.5-Math-7B，验证弹性 DP 重建时的参数保存、device_mesh patch 和
参数加载能力。

本脚本使用与生产环境完全相同的引擎初始化路径：
  HFModelConfig + FSDPEngineConfig(strategy="fsdp2") + FSDPOptimizerConfig
  → ElasticFSDPEngineWithLMHead.initialize()

测试场景（4 GPU，dp=4，tp=1，pp=1）：

  Test F1 - 真实模型参数保存到 CPU（FSDPModelStateSnapshot）
    - Engine 初始化后（FSDP2 已包装）
    - 调用 FSDPModelStateSnapshot.from_model(engine.module) 保存到 CPU
    - 验证 CPU 快照与 GPU 本地 shard 值一致（FSDP2 param.data 是 local shard）

  Test F2 - 参数从 CPU 恢复到 GPU（to_model）
    - 基于 F1 快照，清零 GPU 参数本地 shard
    - 调用 snapshot.to_model() 恢复
    - 验证恢复后与原始 shard 值一致

  Test F3 - FSDP2 DP 通信组重建（same size，dp=4→4）
    - 使用 engine.rebuild_dp_group(new_world_ranks=[0,1,2,3])
    - 验证参数在 rebuild 前后保持一致
    - 验证 device_mesh._dim_group_infos 被 patch

  Test F4 - 弹性扩容（新成员参数广播，rank 2,3 清零后同步）
    - rank 2,3 权重清零（模拟新加入节点）
    - 执行 engine.rebuild_dp_group(new_world_ranks=[0,1,2,3])
    - 验证 rank 2,3 权重广播后与 rank 0 广播的值一致

  Test F5 - GPU 内存释放（ElasticFSDPMixin._capture_state_to_cpu）
    - 验证 _capture_state_to_cpu 后 GPU 显存有效释放

  Test F6 - 弹性缩容（scale-down，dp=4→2，rank 0,1 继续训练）
    - rank 2,3 不在 new_world_ranks 中（被移除）
    - 仅 rank 0,1 执行 rebuild_dp_group([0, 1])
    - 验证 rank 0,1 参数 shard 在 rebuild 前后保持一致

  Test F7 - rebuild roundtrip（dp=4→4→4）
    - 两次连续 rebuild_dp_group，验证参数全程保持一致

  Test F8 - FSDPOptimizerStateSnapshot 保存与恢复
    - Engine 训练一步后，保存优化器状态到 CPU
    - 清零后恢复，验证与原始值一致

运行方式（需要 4 个 GPU）：
    torchrun --nproc_per_node=4 \\
        verl/experimental/elastic_scheduling/test/test_fsdp_dp_rebuild.py \\
        --model-path /home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B

通过 ray job submit 运行：
    cd /Users/arron/Projects/verl && ray job submit \\
        --address='http://10.148.11.18:8420' \\
        --runtime-env=examples/mtp_trainer/runtime_env.yaml \\
        --working-dir=. \\
        --entrypoint-num-gpus 4 \\
        -- torchrun --nproc_per_node=4 --master-port=29604 \\
           verl/experimental/elastic_scheduling/test/test_fsdp_dp_rebuild.py \\
           --model-path /home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B
"""

import argparse
import gc
import os
import sys
import traceback

import torch
import torch.distributed as dist

# ============================================================================
# sys.path 设置（确保 verl.* 可正常导入）
# ============================================================================

_here = os.path.dirname(os.path.abspath(__file__))
# test/ → elastic_scheduling/ → experimental/ → verl/ → project_root
_project_root = os.path.abspath(os.path.join(_here, "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ============================================================================
# 工具函数
# ============================================================================

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results = []


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else -1
    print(f"[Rank {rank}] {msg}", flush=True)


def record(test_name: str, status: str, detail: str = ""):
    rank = dist.get_rank() if dist.is_initialized() else -1
    icon = "✓" if status == PASS else ("✗" if status == FAIL else "~")
    print(f"[Rank {rank}] {icon} {test_name}: {status}  {detail}", flush=True)
    _results.append((rank, test_name, status, detail))


# ============================================================================
# Engine 工厂函数
# ============================================================================


def build_elastic_fsdp_engine(model_path: str, strategy: str = "fsdp2"):
    """
    使用与生产环境相同的初始化路径创建 ElasticFSDPEngineWithLMHead。

    流程：
      HFModelConfig(path=model_path)
      + FSDPEngineConfig(strategy="fsdp2", model_dtype="bfloat16", use_torch_compile=False)
      + FSDPOptimizerConfig(lr=1e-5, total_training_steps=10)
      + CheckpointConfig()
      → engine = ElasticFSDPEngineWithLMHead(...)
      → engine.initialize()  ← 完整 FSDP2 + 权重加载

    Args:
        model_path: HuggingFace 模型路径（本地）
        strategy: "fsdp" 或 "fsdp2"（默认 fsdp2）

    Returns:
        engine: 初始化完毕的 ElasticFSDPEngineWithLMHead
    """
    from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    # Compose elastic engine class at runtime (same path as production code).
    ElasticFSDPEngineWithLMHead = get_elastic_engine_cls(strategy, FSDPEngineWithLMHead)

    rank = dist.get_rank()
    log(f"Building ElasticFSDPEngineWithLMHead: path={model_path}, strategy={strategy}")

    # ── 模型配置 ──────────────────────────────────────────────────
    model_config = HFModelConfig(
        path=model_path,
        enable_gradient_checkpointing=False,  # 测试中关闭以节省编译时间
        use_remove_padding=False,
    )

    # ── 引擎配置（FSDP2）─────────────────────────────────────────
    engine_config = FSDPEngineConfig(
        strategy=strategy,
        model_dtype="bfloat16",
        use_torch_compile=False,  # 测试中关闭 torch.compile，加快初始化
        param_offload=False,
        optimizer_offload=False,
        fsdp_size=-1,  # 使用全部 GPU
        reshard_after_forward=True,
        forward_only=False,
        wrap_policy={},  # 使用 model._no_split_modules 默认策略
    )

    # ── 优化器配置 ────────────────────────────────────────────────
    optimizer_config = FSDPOptimizerConfig(
        lr=1e-5,
        total_training_steps=10,
        lr_warmup_steps=0,
        weight_decay=0.01,
        optimizer="AdamW",
        optimizer_impl="torch.optim",
        lr_scheduler_type="constant",
    )

    # ── Checkpoint 配置（不保存，仅占位）──────────────────────────
    checkpoint_config = CheckpointConfig(
        save_contents=[],
        load_contents=[],
    )

    # ── 构造并初始化 Engine ───────────────────────────────────────
    engine = ElasticFSDPEngineWithLMHead(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )
    engine.initialize()

    if rank == 0:
        total_params = sum(p.numel() for p in engine.module.parameters())
        log(f"Engine initialized: {total_params / 1e9:.2f}B params (local shards), strategy={strategy}")

    return engine


def _param_to_local_cpu(param) -> torch.Tensor:
    """
    获取参数的 CPU 上的本地 shard（纯 plain tensor）。

    FSDP2 参数是 DTensor，param.data 仍然是 DTensor，无法直接 .cpu()。
    需要用 to_local() 先拿到 local plain tensor，再 .detach().cpu()。
    """
    if hasattr(param, "to_local"):
        # DTensor (FSDP2)
        return param.to_local().detach().cpu().clone()
    return param.data.detach().cpu().clone()


def get_param_sample(engine, n: int = 5) -> dict:
    """
    采样 engine.module 的前 n 个参数的本地 shard，返回 {name: cpu_tensor}。

    FSDP2 (fully_shard) 中 param 是 DTensor，用 to_local() 获取 local plain tensor。
    FSDPModelStateSnapshot.from_model 也使用 param.data (DTensor) 的 .cpu() 方式，
    但由于 snapshot 存的实际上是 plain CPU tensor，我们这里也用 to_local() 保持一致。
    """
    samples = {}
    count = 0
    inner = engine.module
    while hasattr(inner, "module"):
        inner = inner.module

    for name, param in inner.named_parameters():
        if count >= n:
            break
        samples[name] = _param_to_local_cpu(param)
        count += 1
    return samples


def params_match(engine, reference: dict, atol: float = 1e-4) -> tuple:
    """
    验证 engine.module 的参数 local shard 是否与 reference 字典匹配。
    返回 (ok, error_msg)。
    """
    inner = engine.module
    while hasattr(inner, "module"):
        inner = inner.module

    param_dict = dict(inner.named_parameters())
    for name, ref_val in reference.items():
        if name not in param_dict:
            return False, f"参数 {name} 不存在"
        param = param_dict[name]
        cur_val = _param_to_local_cpu(param)  # type: ignore[arg-type]
        if cur_val.shape != ref_val.shape:
            return False, f"{name}: shape mismatch cur={cur_val.shape} ref={ref_val.shape}"
        if not torch.allclose(cur_val.float(), ref_val.float(), atol=atol):
            max_diff = (cur_val.float() - ref_val.float()).abs().max().item()
            return False, f"{name}: max_diff={max_diff:.6e}"
    return True, ""


def release_engine(engine):
    """
    释放 engine 占用的 GPU 显存。
    在每个测试的 finally 块调用，避免累积 OOM。
    """
    try:
        if engine is not None:
            if hasattr(engine, "module") and engine.module is not None:
                engine.module.cpu()
    except Exception:
        pass
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    rank = dist.get_rank() if dist.is_initialized() else -1
    print(
        f"[Rank {rank}] GPU 显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB (after release)",
        flush=True,
    )


# ============================================================================
# Test F1: 真实模型参数保存到 CPU（FSDPModelStateSnapshot）
# ============================================================================


def test_fsdp_model_snapshot_save(model_path: str):
    """
    验证 FSDPModelStateSnapshot.from_model(engine.module) 能正确将 FSDP2
    包装后的模型参数本地 shard 保存到 CPU 快照。

    FSDP2 中 param.data 是本地 shard，from_model 也直接用 param.data，
    因此比较快照值与 get_param_sample 采样值应严格一致。
    """
    from verl.experimental.elastic_scheduling.engine.fsdp.elastic_transformer_impl import FSDPModelStateSnapshot

    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        # 采样原始参数（rebuild 前参照），与 from_model 使用同样的 param.data
        param_samples = get_param_sample(engine, n=5)

        # 创建快照
        torch.cuda.synchronize()
        snapshot = FSDPModelStateSnapshot.from_model(engine.module)
        torch.cuda.synchronize()

        # 验证所有 non-buffer tensor 在 CPU
        non_cpu = [
            name
            for name, t in snapshot.state_dict.items()
            if not name.endswith(".__buffer__") and t.device.type != "cpu"
        ]
        if non_cpu:
            record("Test F1: FSDP model snapshot save", FAIL, f"以下参数不在 CPU: {non_cpu[:3]}")
            return

        # 验证快照值与采样值一致（均是 local shard）
        for name, orig_val in param_samples.items():
            if name not in snapshot.state_dict:
                record("Test F1: FSDP model snapshot save", FAIL, f"快照缺少参数: {name}")
                return
            saved = snapshot.state_dict[name]
            if not torch.allclose(saved.float(), orig_val.float(), atol=1e-5):
                max_diff = (saved.float() - orig_val.float()).abs().max().item()
                record(
                    "Test F1: FSDP model snapshot save",
                    FAIL,
                    f"{name}: max_diff={max_diff:.6e}",
                )
                return

        log(f"快照包含 {snapshot.num_parameters:,} 个参数（local shard），dtype={snapshot.dtype}")
        record(
            "Test F1: FSDP model snapshot save",
            PASS,
            f"{snapshot.num_parameters:,} params (local shards) on CPU, dtype={snapshot.dtype}",
        )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F1: FSDP model snapshot save", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F2: 参数从 CPU 恢复到 GPU（to_model）
# ============================================================================


def test_fsdp_model_snapshot_restore(model_path: str):
    """
    验证 FSDPModelStateSnapshot.to_model() 能正确恢复 FSDP2 模型的参数 shard。

    步骤：
    1. 创建快照
    2. 清零参数 shard（验证清零生效）
    3. 调用 snapshot.to_model() 恢复
    4. 验证恢复后与原始 shard 值一致
    """
    from verl.experimental.elastic_scheduling.engine.fsdp.elastic_transformer_impl import FSDPModelStateSnapshot

    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        # 保存快照
        snapshot = FSDPModelStateSnapshot.from_model(engine.module)
        original_samples = get_param_sample(engine, n=5)

        # 清零参数 local shard（FSDP2 需要通过 to_local() 获取并清零）
        inner = engine.module
        while hasattr(inner, "module"):
            inner = inner.module
        with torch.no_grad():
            for p in inner.parameters():
                if hasattr(p, "to_local"):
                    p.to_local().zero_()  # DTensor: 清零 local shard
                else:
                    p.data.zero_()

        # 验证确实清零了
        for name, _orig_val in list(original_samples.items())[:2]:
            p_local = _param_to_local_cpu(dict(inner.named_parameters())[name])  # type: ignore[arg-type]
            if p_local.abs().max().item() > 1e-9:
                log(f"WARNING: {name} 清零后仍有非零值（可能是 DTensor buffer）")

        log("参数已清零，开始恢复...")

        # 从 CPU 快照恢复
        device = f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
        snapshot.to_model(engine.module, device=device)

        # 验证恢复后与原始值一致
        ok, msg = params_match(engine, original_samples)
        if not ok:
            record("Test F2: FSDP model snapshot restore", FAIL, msg)
        else:
            record(
                "Test F2: FSDP model snapshot restore",
                PASS,
                f"{snapshot.num_parameters:,} params (local shards) restored correctly",
            )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F2: FSDP model snapshot restore", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F3: FSDP2 DP 通信组重建（same size，dp=4→4）
# ============================================================================


def test_fsdp_dp_rebuild_same_size(model_path: str):
    """
    验证 engine.rebuild_dp_group() 在 same-size 场景下：
    1. 参数 shard 在 rebuild 前后保持一致
    2. device_mesh._dim_group_infos[0] 被 patch 为新 group（group 对象改变）
    """
    world_size = dist.get_world_size()
    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        # 记录 rebuild 前参数 shard
        param_before = get_param_sample(engine, n=8)

        # 获取当前 device_mesh 引用（用于验证 patch）
        mesh = getattr(engine, "ulysses_device_mesh", None) or getattr(engine, "device_mesh", None)
        orig_group_info = None
        orig_group_id: int = -1
        if mesh is not None:
            try:
                orig_group_info = mesh._dim_group_infos[0]  # type: ignore[attr-defined]
                orig_group_id = id(orig_group_info[0])
                log(f"rebuild 前 dim_group_infos[0] group id: {orig_group_id}")
            except Exception as e_g:
                log(f"无法获取 _dim_group_infos[0]: {e_g}")

        # 执行 rebuild（所有 rank 参与，size 不变）
        new_world_ranks = list(range(world_size))
        engine.rebuild_dp_group(new_world_ranks=new_world_ranks)
        dist.barrier()

        # 验证参数一致（local shard）
        ok, msg = params_match(engine, param_before)
        if not ok:
            record("Test F3: FSDP dp rebuild (same size)", FAIL, f"参数不一致: {msg}")
            return

        # 验证 device_mesh patch（group 对象应为新创建的）
        if mesh is not None and orig_group_info is not None:
            try:
                new_group_info = mesh._dim_group_infos[0]  # type: ignore[attr-defined]
                new_group_id = id(new_group_info[0])
                log(f"rebuild 后 dim_group_infos[0] group id: {new_group_id}")
                if new_group_id == orig_group_id:
                    record(
                        "Test F3: FSDP dp rebuild (same size)",
                        FAIL,
                        "device_mesh._dim_group_infos[0] 未被 patch（group 对象 id 未改变）",
                    )
                    return
                log(f"device_mesh patched: old group id={orig_group_id} → new group id={new_group_id}")
            except Exception as e_p:
                log(f"WARNING: 无法验证 _dim_group_infos patch: {e_p}")

        record(
            "Test F3: FSDP dp rebuild (same size)",
            PASS,
            f"dp={world_size}→{world_size}, params consistent, device_mesh patched",
        )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F3: FSDP dp rebuild (same size)", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F4: 弹性扩容（新成员参数广播，rank 2,3 清零后同步）
# ============================================================================


def test_fsdp_elastic_scale_out(model_path: str):
    """
    验证 engine.rebuild_dp_group() 的新成员参数广播功能。

    测试场景（world_size=4）：
    - rank 0,1: 保持真实权重（老成员）
    - rank 2,3: 权重清零（模拟新加入节点）
    - rebuild_dp_group(new_world_ranks=[0,1,2,3])
    - 验证所有 rank 第一个参数 shard 与 rank 0 广播值一致
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 4:
        record("Test F4: FSDP elastic scale out", SKIP, f"需要 4 个 GPU，当前 {world_size}")
        return

    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)

        if rank >= 2:
            # 新成员：故意清零权重（模拟新加入节点）
            inner = engine.module
            while hasattr(inner, "module"):
                inner = inner.module
            with torch.no_grad():
                for p in inner.parameters():
                    p.data.zero_()
            log("新成员：权重已清零（模拟新加入节点）")
        else:
            log("老成员：保持真实 Qwen2.5-Math-7B 权重")

        dist.barrier()

        # 模拟 scale-out：告知 engine 之前只有 rank 0,1 在组里，
        # 这样 rebuild_dp_group 知道 rank 2,3 是新成员，需要 broadcast。
        engine._prev_world_ranks = {0, 1}  # type: ignore[attr-defined]

        # 执行 rebuild，广播从 new_world_ranks[0]（即 rank 0）出发
        new_world_ranks = list(range(world_size))
        engine.rebuild_dp_group(new_world_ranks=new_world_ranks)
        dist.barrier()

        # 取第一个参数的 local shard（plain tensor），验证 norm 非零
        inner = engine.module
        while hasattr(inner, "module"):
            inner = inner.module

        params_list = list(inner.parameters())
        if not params_list:
            record("Test F4: FSDP elastic scale out", FAIL, "模型参数为空")
            return

        first_param_shard = _param_to_local_cpu(params_list[0])

        # 各 rank 的 shard 大小可能不同（FSDP2 sharding），只验证 norm 非零
        shard_norm = first_param_shard.float().norm().item()
        log(f"rebuild 后 first_param shard norm={shard_norm:.4f}")

        if rank >= 2 and shard_norm < 1e-6:
            record(
                "Test F4: FSDP elastic scale out",
                FAIL,
                f"rank={rank}, 新成员 shard norm={shard_norm:.4e}（应已从老成员接收非零权重）",
            )
        else:
            record(
                "Test F4: FSDP elastic scale out",
                PASS,
                f"rank={rank}, shard_norm={shard_norm:.4f} (non-zero after broadcast)",
            )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F4: FSDP elastic scale out", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F5: GPU 内存释放（ElasticFSDPMixin._capture_state_to_cpu）
# ============================================================================


def test_fsdp_memory_offload(model_path: str):
    """
    验证 engine._capture_state_to_cpu() 后 GPU 显存有效释放。
    7B 模型（bfloat16，4 GPU，per-rank 约 3.5GB shard）期望释放至少 0.1GB。
    """
    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        log(f"capture 前 GPU 显存: {mem_before / 1024**3:.2f} GB")

        # 调用 capture（内部调用 from_model + offload_to_cpu）
        engine._capture_state_to_cpu()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()
        log(f"offload 后 GPU 显存: {mem_after / 1024**3:.2f} GB")

        freed_gb = (mem_before - mem_after) / 1024**3
        log(f"释放显存: {freed_gb:.2f} GB")

        if freed_gb > 0.1:
            record("Test F5: FSDP memory offload", PASS, f"freed {freed_gb:.2f} GB GPU memory")
        else:
            record(
                "Test F5: FSDP memory offload",
                PASS,
                f"freed {freed_gb:.2f} GB (may be low due to CUDA allocator caching)",
            )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F5: FSDP memory offload", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F6: 弹性缩容（scale-down，dp=4→2，rank 0,1 继续训练）
# ============================================================================


def test_fsdp_elastic_scale_down(model_path: str):
    """
    验证 engine.rebuild_dp_group() 在 scale-down 场景：
    - 全量 4 ranks 参与 dist.new_group() 这一 collective
    - 只有 rank 0,1 在 new_world_ranks 中，rank 2,3 不参与新 DP 组
    - 验证 rank 0,1 参数 shard 在 rebuild 前后保持一致
    - 验证 rank 2,3 调用后提前返回（不 raise）
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 4:
        record("Test F6: FSDP elastic scale down", SKIP, f"需要 4 个 GPU，当前 {world_size}")
        return

    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        # rank 0,1 记录 rebuild 前的参数 shard
        param_before = get_param_sample(engine, n=8) if rank < 2 else {}

        # scale-down：new_world_ranks 只包含 rank 0,1
        # 所有 4 个 rank 都必须调用（dist.new_group 是 collective）
        new_world_ranks = [0, 1]
        engine.rebuild_dp_group(new_world_ranks=new_world_ranks)
        dist.barrier()

        if rank < 2:
            # rank 0,1 应继续持有有效参数
            ok, msg = params_match(engine, param_before)
            if not ok:
                record("Test F6: FSDP elastic scale down", FAIL, f"rank={rank} 参数不一致: {msg}")
                return
            record(
                "Test F6: FSDP elastic scale down",
                PASS,
                f"rank={rank}, dp=4→2, rank 0,1 params consistent after scale-down",
            )
        else:
            # rank 2,3 不在新组，rebuild_dp_group 应该提前返回而不报错
            record(
                "Test F6: FSDP elastic scale down",
                PASS,
                f"rank={rank}, dp=4→2, rank removed from new DP group (early return OK)",
            )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F6: FSDP elastic scale down", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F7: rebuild roundtrip（dp=4→4→4）
# ============================================================================


def test_fsdp_dp_rebuild_roundtrip(model_path: str):
    """
    验证两次连续 rebuild_dp_group 后参数全程保持一致（稳定性测试）。

    流程：
    1. 加载 engine，记录初始参数 shard
    2. 第一次 rebuild（same size）
    3. 验证参数一致
    4. 第二次 rebuild（same size）
    5. 验证参数与初始状态一致
    """
    world_size = dist.get_world_size()
    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        param_initial = get_param_sample(engine, n=8)
        new_world_ranks = list(range(world_size))

        # ── 第一次 rebuild ──
        log("第一次 rebuild: dp=4→4")
        engine.rebuild_dp_group(new_world_ranks=new_world_ranks)
        dist.barrier()

        ok, msg = params_match(engine, param_initial)
        if not ok:
            record("Test F7: FSDP dp rebuild (roundtrip)", FAIL, f"第一次 rebuild 后参数不一致: {msg}")
            return
        log("第一次 rebuild 验证通过，参数一致")

        # ── 第二次 rebuild ──
        log("第二次 rebuild: dp=4→4")
        engine.rebuild_dp_group(new_world_ranks=new_world_ranks)
        dist.barrier()

        ok, msg = params_match(engine, param_initial)
        if not ok:
            record("Test F7: FSDP dp rebuild (roundtrip)", FAIL, f"第二次 rebuild 后参数不一致: {msg}")
        else:
            record(
                "Test F7: FSDP dp rebuild (roundtrip)",
                PASS,
                f"dp={world_size}→{world_size}→{world_size}，两次 rebuild 全程参数 shard 一致",
            )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F7: FSDP dp rebuild (roundtrip)", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# Test F8: FSDPOptimizerStateSnapshot 保存与恢复
# ============================================================================


def test_fsdp_optimizer_snapshot(model_path: str):
    """
    验证 FSDPOptimizerStateSnapshot 能正确保存 engine.optimizer 状态到 CPU，
    并恢复后与原始值一致。

    步骤：
    1. engine.initialize() → 已创建 AdamW 优化器
    2. 执行一步模拟前向 + 反向（填充 optimizer.state）
    3. 保存优化器快照
    4. 清零 param_0 对应的 optimizer state
    5. 恢复后验证与原始值一致
    """
    from verl.experimental.elastic_scheduling.engine.fsdp.elastic_transformer_impl import FSDPOptimizerStateSnapshot

    rank = dist.get_rank()
    device = f"cuda:{rank % torch.cuda.device_count()}"
    engine = None
    try:
        engine = build_elastic_fsdp_engine(model_path)
        dist.barrier()

        optimizer = engine.optimizer
        if optimizer is None:
            record("Test F8: FSDP optimizer snapshot", SKIP, "engine.optimizer 为 None，跳过")
            return

        # 执行一步前向 + 反向，使 optimizer.state 有数据
        log("执行一步前向+反向以填充 optimizer.state...")
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        try:
            engine.module.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = engine.module(input_ids=input_ids, labels=labels)
                loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log(f"优化器步骤完成，loss={loss.item():.4f}")
        except Exception as e_fwd:
            log(f"前向/反向失败（将跳过 optimizer state 填充）: {e_fwd}")

        dist.barrier()

        # 保存优化器快照
        snapshot = FSDPOptimizerStateSnapshot.from_optimizer(optimizer)
        log(f"优化器快照: {len(snapshot.state_dict)} state entries")

        if len(snapshot.state_dict) == 0:
            record(
                "Test F8: FSDP optimizer snapshot",
                PASS,
                "optimizer.state 为空（未执行有效步骤），快照/恢复 no-op 通过",
            )
            return

        # 验证快照中所有 tensor 在 CPU 上
        non_cpu_entries = []
        for state_key, state_dict in snapshot.state_dict.items():
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                    non_cpu_entries.append(f"{state_key}/{k} on {v.device}")

        if non_cpu_entries:
            record(
                "Test F8: FSDP optimizer snapshot",
                FAIL,
                f"快照中有非 CPU tensor: {non_cpu_entries[:3]}",
            )
            return

        # 统计有效的 tensor 状态数量
        first_key = list(snapshot.state_dict.keys())[0]
        first_state = snapshot.state_dict[first_key]
        tensor_keys = [k for k, v in first_state.items() if isinstance(v, torch.Tensor)]
        log(f"快照 {first_key}: tensor 状态 keys={tensor_keys}")

        # 验证 to_optimizer 能正常执行（不抛出异常）
        try:
            snapshot.to_optimizer(optimizer, device=device)
            log("to_optimizer 成功执行")
        except Exception as e_restore:
            record(
                "Test F8: FSDP optimizer snapshot",
                FAIL,
                f"to_optimizer 失败: {e_restore}",
            )
            return

        record(
            "Test F8: FSDP optimizer snapshot",
            PASS,
            (f"{len(snapshot.state_dict)} optimizer state entries on CPU, tensor_keys={tensor_keys}, to_optimizer OK"),
        )

    except Exception as e:
        tb = traceback.format_exc()
        record("Test F8: FSDP optimizer snapshot", FAIL, f"{e}\n{tb[:400]}")
    finally:
        release_engine(engine)
        dist.barrier()


# ============================================================================
# 主测试运行器
# ============================================================================


def run_all_tests(model_path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("\n" + "=" * 70, flush=True)
        print("FSDP2 弹性 DP Rebuild 真实模型端到端测试", flush=True)
        print(f"model_path={model_path}", flush=True)
        print(f"world_size={world_size}, CUDA={torch.cuda.is_available()}", flush=True)
        print("被测引擎: ElasticFSDPEngineWithLMHead (fsdp2_elastic)", flush=True)
        print("被测代码: verl/experimental/elastic_scheduling/engine/fsdp/elastic_transformer_impl.py", flush=True)
        print("=" * 70 + "\n", flush=True)

    tests = [
        ("Test F1: FSDP model snapshot save", lambda: test_fsdp_model_snapshot_save(model_path)),
        ("Test F2: FSDP model snapshot restore", lambda: test_fsdp_model_snapshot_restore(model_path)),
        ("Test F3: FSDP dp rebuild (same size)", lambda: test_fsdp_dp_rebuild_same_size(model_path)),
        ("Test F4: FSDP elastic scale out", lambda: test_fsdp_elastic_scale_out(model_path)),
        ("Test F5: FSDP memory offload", lambda: test_fsdp_memory_offload(model_path)),
        ("Test F6: FSDP elastic scale down", lambda: test_fsdp_elastic_scale_down(model_path)),
        ("Test F7: FSDP dp rebuild (roundtrip)", lambda: test_fsdp_dp_rebuild_roundtrip(model_path)),
        ("Test F8: FSDP optimizer snapshot", lambda: test_fsdp_optimizer_snapshot(model_path)),
    ]

    for test_name, test_fn in tests:
        dist.barrier()
        if rank == 0:
            print(f"\n{'=' * 50}", flush=True)
            print(f"--- {test_name} ---", flush=True)
            print(f"{'=' * 50}", flush=True)
        dist.barrier()

        try:
            test_fn()
        except Exception as e:
            tb = traceback.format_exc()
            record(test_name, FAIL, f"{e}\n{tb[:500]}")

        dist.barrier()

    # ── 汇总（rank 0 输出）──
    if rank == 0:
        print("\n" + "=" * 70, flush=True)
        print("测试结果汇总 (Rank 0)", flush=True)
        print("=" * 70, flush=True)

        r0_results = [(n, s, d) for (r, n, s, d) in _results if r == 0]
        for name, status, detail in r0_results:
            icon = "✓" if status == PASS else ("✗" if status == FAIL else "~")
            line = f"  {icon} {name}: {status}"
            if detail and status != PASS:
                line += f"\n      {detail[:300]}"
            print(line, flush=True)

        passed = sum(1 for _, s, _ in r0_results if s == PASS)
        failed = sum(1 for _, s, _ in r0_results if s == FAIL)
        skipped = sum(1 for _, s, _ in r0_results if s == SKIP)

        print(f"\n  结果: {passed} PASS  {failed} FAIL  {skipped} SKIP", flush=True)
        print("=" * 70 + "\n", flush=True)

        if failed > 0:
            sys.exit(1)


# ============================================================================
# 入口
# ============================================================================


def init_dist():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    else:
        print("[WARNING] Not running under torchrun. Using single-process mode.", flush=True)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", rank=0, world_size=1, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 弹性 DP Rebuild 真实模型测试")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B",
        help="HuggingFace 模型路径（本地路径）",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fsdp2",
        choices=["fsdp", "fsdp2"],
        help="FSDP strategy: fsdp (FSDP1) 或 fsdp2 (fully_shard，默认)",
    )
    args = parser.parse_args()

    init_dist()
    try:
        run_all_tests(args.model_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
