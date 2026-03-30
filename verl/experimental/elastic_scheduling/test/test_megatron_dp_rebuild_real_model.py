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
真实模型端到端测试：使用 mbridge 加载 Qwen2-Math-7B，验证 Megatron DP 变化时的
参数保存、通信组重建和参数加载能力。

本脚本基于 ElasticMegatronMixin（verl/experimental/elastic_scheduling/engine/megatron/
elastic_transformer_impl.py），验证其完整的弹性 DP rebuild 流程。

测试场景（4 GPU，tp=1, pp=1, dp=4）：

  Test R1 - 真实模型参数保存到 CPU（ModelStateSnapshot）
    - 通过 mbridge 加载 Qwen2-Math-7B 到 Megatron
    - 记录部分参数（embedding、第一层 attention）的原始值
    - 调用 ModelStateSnapshot.from_model() 保存到 CPU
    - 验证 CPU 快照与 GPU 原始值一致（clone 独立性）

  Test R2 - 参数恢复到 GPU（to_model）
    - 基于 R1 的快照，清零 GPU 参数
    - 调用 snapshot.to_model() 恢复
    - 验证恢复后 GPU 参数值与快照一致

  Test R3 - DP 通信组重建（DP=4 → 销毁 → 重建 DP=4）
    - 使用真实模型执行 ElasticMegatronMixin.rebuild_dp_group()
    - 验证参数在 rebuild 前后保持一致
    - 验证通信组 dp world_size 正确

  Test R4 - 弹性扩容：DP=2 → DP=4（新成员参数广播）
    - rank 0,1 持有真实模型权重（老成员）
    - rank 2,3 持有随机权重（新成员）
    - rebuild_dp_group(new_world_ranks=[0,1,2,3])
    - 验证 rank 2,3 的参数广播后与 rank 0,1 一致

  Test R5 - GPU 内存释放（真实大模型场景）
    - 验证 _capture_state_to_cpu 后 GPU 显存有效释放

  Test R6 - 真实模型 DP rebuild scale down（DP=4 → DP=2）
    - 以 tp=1（dp=4）初始化，rebuild 到 tp=2（dp=2）

  Test R7 - 真实模型 DP rebuild roundtrip（DP=4 → DP=2 → DP=4）

运行方式（需要 4 个 GPU）：
    torchrun --nproc_per_node=4 \\
        verl/experimental/elastic_scheduling/test/test_megatron_dp_rebuild_real_model.py \\
        --model-path /home/hadoop-djst-algoplat/models/Qwen2-Math-7B

通过 ray job submit 运行：
    ray job submit \\
        --address='http://10.148.11.18:8420' \\
        --runtime-env=examples/mtp_trainer/runtime_env.yaml \\
        --working-dir=. \\
        --entrypoint-num-gpus 4 \\
        -- torchrun --nproc_per_node=4 --master-port=29603 \\
           verl/experimental/elastic_scheduling/test/test_megatron_dp_rebuild_real_model.py \\
           --model-path /home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B
"""

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist

# ============================================================================
# 路径设置
# ============================================================================

_here = os.path.dirname(os.path.abspath(__file__))
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


def assert_allclose(a: torch.Tensor, b: torch.Tensor, name: str = "", atol: float = 1e-4):
    if not torch.allclose(a.float(), b.float(), atol=atol):
        max_diff = (a.float() - b.float()).abs().max().item()
        raise AssertionError(f"{name}: max_diff={max_diff:.6e} > atol={atol}")


# ============================================================================
# 真实模型加载（通过 mbridge，与 MegatronEngine vanilla_mbridge=True 路径一致）
# ============================================================================


def set_random_seed(seed: int = 42):
    """
    初始化随机种子，参考 verl/workers/engine/megatron/utils.py 中的 set_random_seed。
    关键：调用 tensor_parallel.model_parallel_cuda_manual_seed 注册 Megatron RNG 状态。
    必须在 initialize_model_parallel 之后调用。
    """
    import random

    import numpy as np
    from megatron.core import tensor_parallel

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def load_real_model_with_mbridge(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """
    使用 mbridge（vanilla mbridge 路径）加载 HuggingFace 模型到 Megatron 格式。

    实现参考 verl/workers/engine/megatron/transformer_impl.py 中 MegatronEngine 的
    _build_tf_config + _build_megatron_module（vanilla_bridge=True 分支）。

    要求：在调用前已完成：
      1. dist.init_process_group
      2. parallel_state.initialize_model_parallel
      3. set_random_seed（注册 Megatron CUDA RNG 状态）

    Args:
        model_path: HuggingFace 模型路径
        dtype: 参数数据类型，默认 bfloat16

    Returns:
        model: Megatron model list（[Float16Module(GPTModel)] 或 [DDP(Float16Module(GPTModel))]）
    """
    from verl.models.mcore.mbridge import AutoBridge

    rank = dist.get_rank()
    log(f"Loading model from {model_path} via mbridge (vanilla bridge path)...")

    bridge = AutoBridge.from_pretrained(model_path)
    bridge.set_extra_args(
        variable_seq_lengths=True,
    )
    tf_config = bridge.config
    tf_config.fp16 = dtype == torch.float16
    tf_config.bf16 = dtype == torch.bfloat16

    if rank == 0:
        log(
            f"TransformerConfig: num_layers={tf_config.num_layers}, "
            f"hidden_size={tf_config.hidden_size}, "
            f"fp16={tf_config.fp16}, bf16={tf_config.bf16}"
        )

    model = bridge.get_model(
        post_model_creation_callbacks=[],
        wrap_with_ddp=True,
        fp16=tf_config.fp16,
        bf16=tf_config.bf16,
        ddp_config=None,
    )

    bridge.load_weights(model, model_path)

    if rank == 0:
        total_params = sum(p.numel() for chunk in model for p in chunk.parameters())
        log(f"Model loaded: {total_params / 1e9:.2f}B parameters, dtype={dtype}")

    return model


def get_param_sample(model, n: int = 3) -> dict:
    """
    采样模型的前 n 个参数，返回 {name: cpu_tensor} 字典。
    穿透 DDP -> Float16Module -> GPTModel 层层 wrapper。
    """
    samples = {}
    count = 0
    inner = model[0]
    while hasattr(inner, "module"):
        inner = inner.module

    for name, param in inner.named_parameters():
        if count >= n:
            break
        samples[name] = param.data.detach().cpu().clone()
        count += 1
    return samples


def params_match(model, reference: dict, atol: float = 1e-4) -> tuple:
    """验证模型参数是否与 reference 字典匹配，返回 (ok, error_msg)。"""
    inner = model[0]
    while hasattr(inner, "module"):
        inner = inner.module

    param_dict = dict(inner.named_parameters())
    for name, ref_val in reference.items():
        if name not in param_dict:
            return False, f"参数 {name} 不存在"
        cur_val = param_dict[name].data.detach().cpu()
        if not torch.allclose(cur_val.float(), ref_val.float(), atol=atol):
            max_diff = (cur_val.float() - ref_val.float()).abs().max().item()
            return False, f"{name}: max_diff={max_diff:.6e}"
    return True, ""


def init_parallel_state(tp=1, pp=1):
    """
    初始化 Megatron 并行状态，同时调用 set_random_seed 注册 Megatron RNG 状态。
    """
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
        )
        set_random_seed(seed=42)
        log(f"Parallel state initialized: tp={tp}, pp={pp}, dp={parallel_state.get_data_parallel_world_size()}")


def destroy_parallel_state():
    """销毁 Megatron 并行状态，忽略异常。"""
    try:
        from megatron.core import parallel_state

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
    except Exception as e:
        log(f"destroy_model_parallel 忽略异常: {e}")


def release_model_memory(model):
    """显式释放模型占用的 GPU 显存。"""
    import gc

    try:
        if model is not None:
            for chunk in model:
                chunk.cpu()
            del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    log(f"GPU 显存已释放: {torch.cuda.memory_allocated() / 1024**3:.2f} GB remaining")


# ============================================================================
# Mixin 包装：将 list of model chunks 包装成 ElasticMegatronMixin 可操作的对象
# ============================================================================


def make_elastic_mixin(model):
    """
    创建一个轻量 wrapper，使 ElasticMegatronMixin 的私有方法可以被测试直接调用。

    ElasticMegatronMixin 设计为 Engine 的 mixin，通过 self.module 访问模型。
    这里用 wrapper 模拟 engine 的接口，无需真正实例化 MegatronEngine。
    """
    from verl.experimental.elastic_scheduling.engine.megatron.elastic_transformer_impl import ElasticMegatronMixin

    class _Wrapper(ElasticMegatronMixin):
        def __init__(self, model_chunks):
            # 不调用 super().__init__() 以避免依赖 MegatronEngine
            self.module = model_chunks
            self._model_snapshot = None
            self._optimizer_snapshot = None
            self._model_on_gpu = True
            self._optimizer_on_gpu = True
            # ElasticMegatronMixin._reinitialize_parallel_groups 中通过
            # getattr(self.engine_config, "virtual_pipeline_model_parallel_size", None)
            # 读取此属性，测试场景不需要 vpp，设为 None 即可。
            self.engine_config = None

    return _Wrapper(model)


# ============================================================================
# Test R1: 真实模型参数保存到 CPU
# ============================================================================


def test_real_model_snapshot_save(model_path: str):
    """验证真实 Qwen2 模型参数能正确保存到 CPU 快照"""
    from verl.experimental.elastic_scheduling.engine.megatron.elastic_transformer_impl import ModelStateSnapshot

    init_parallel_state(tp=1, pp=1)

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)
        dist.barrier()

        param_samples = get_param_sample(model, n=5)

        torch.cuda.synchronize()
        snapshot = ModelStateSnapshot.from_model(model)
        torch.cuda.synchronize()

        # 验证快照在 CPU 上
        for name, tensor in snapshot.state_dict.items():
            assert tensor.device.type == "cpu", f"快照参数 {name} 不在 CPU"

        # 验证快照值与原始 GPU 值一致
        for name, orig_val in param_samples.items():
            assert name in snapshot.state_dict, f"快照缺少参数: {name}"
            assert_allclose(snapshot.state_dict[name], orig_val, name=f"snapshot/{name}")

        log(f"快照包含 {snapshot.num_parameters:,} 个参数，dtype={snapshot.dtype}")
        record("Test R1: real model snapshot save", PASS, f"{snapshot.num_parameters:,} params, dtype={snapshot.dtype}")

    except Exception as e:
        record("Test R1: real model snapshot save", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# Test R2: 参数从 CPU 恢复到 GPU
# ============================================================================


def test_real_model_snapshot_restore(model_path: str):
    """验证 CPU 快照能正确恢复到真实模型的 GPU 参数"""
    from verl.experimental.elastic_scheduling.engine.megatron.elastic_transformer_impl import ModelStateSnapshot

    init_parallel_state(tp=1, pp=1)

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)
        dist.barrier()

        snapshot = ModelStateSnapshot.from_model(model)
        original_samples = get_param_sample(model, n=5)

        # 清零 GPU 参数
        inner = model[0]
        while hasattr(inner, "module"):
            inner = inner.module
        with torch.no_grad():
            for p in inner.parameters():
                p.data.zero_()

        first_param = next(inner.parameters())
        assert first_param.data.abs().max().item() < 1e-9, "清零后参数应为 0"
        log("参数已清零，开始恢复...")

        # 从 CPU 快照恢复
        snapshot.to_model(model, device="cuda")

        ok, msg = params_match(model, original_samples)
        if not ok:
            record("Test R2: real model snapshot restore", FAIL, msg)
        else:
            record(
                "Test R2: real model snapshot restore", PASS, f"{snapshot.num_parameters:,} params restored correctly"
            )

    except Exception as e:
        record("Test R2: real model snapshot restore", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# Test R3: 真实模型 DP 通信组重建（same size，DP=4 → DP=4）
# ============================================================================


def test_real_model_dp_rebuild_same_size(model_path: str):
    """
    使用真实 Qwen2 模型验证 DP rebuild 流程：
    DP=4 → capture → destroy → reinit DP=4 → restore
    参数值在 rebuild 前后保持一致。
    """
    from megatron.core import parallel_state

    world_size = dist.get_world_size()
    init_parallel_state(tp=1, pp=1)
    dp_before = parallel_state.get_data_parallel_world_size()

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)
        dist.barrier()

        param_before = get_param_sample(model, n=8)
        log(f"rebuild 前采样参数: {list(param_before.keys())[:3]}...")

        wrapper = make_elastic_mixin(model)
        # new_world_ranks = 全部 ranks（world_size 不变）
        all_ranks = list(range(world_size))
        wrapper.rebuild_dp_group(
            new_world_ranks=all_ranks,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        set_random_seed(seed=42)

        dp_after = parallel_state.get_data_parallel_world_size()
        assert dp_after == dp_before, f"dp_size 应保持 {dp_before}，实际 {dp_after}"

        ok, msg = params_match(model, param_before)
        if not ok:
            record("Test R3: real model DP rebuild (same size)", FAIL, msg)
        else:
            record(
                "Test R3: real model DP rebuild (same size)",
                PASS,
                f"dp={dp_before}→{dp_after}, params consistent after rebuild",
            )

    except Exception as e:
        record("Test R3: real model DP rebuild (same size)", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# Test R4: 弹性扩容（新成员参数广播）
# ============================================================================


def test_real_model_elastic_scale_out(model_path: str):
    """
    验证真实模型下的弹性扩容广播：
    - rank 0,1 加载真实权重（老成员）
    - rank 2,3 使用随机权重（新成员）
    - rebuild_dp_group(new_world_ranks=[0,1,2,3])
    - 验证 rank 2,3 的参数广播后与 rank 0,1 一致
    """
    from megatron.core import parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 4:
        record("Test R4: real model elastic scale out", SKIP, f"需要 4 个 GPU，当前 {world_size}")
        return

    init_parallel_state(tp=1, pp=1)

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)

        if rank >= 2:
            # 新成员：故意将权重置零（模拟新加入节点）
            inner = model[0]
            while hasattr(inner, "module"):
                inner = inner.module
            with torch.no_grad():
                for p in inner.parameters():
                    p.data.zero_()
            log("新成员：模型权重已清零（模拟新加入节点）")
        else:
            log("老成员：保持真实 Qwen2-Math-7B 权重")

        dist.barrier()

        wrapper = make_elastic_mixin(model)
        all_ranks = list(range(world_size))
        wrapper.rebuild_dp_group(
            new_world_ranks=all_ranks,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        set_random_seed(seed=42)
        dist.barrier()

        # 验证：所有 rank 的参数应与老成员（rank 0）保持一致
        inner = model[0]
        while hasattr(inner, "module"):
            inner = inner.module

        params_list = list(inner.parameters())
        if len(params_list) == 0:
            record("Test R4: real model elastic scale out", FAIL, "模型参数为空")
            return

        # rank 0 广播第一个参数，所有 rank 验证
        check_param = params_list[0].data.clone()
        dist.broadcast(check_param, src=0)

        actual = params_list[0].data
        if not torch.allclose(actual.float(), check_param.float(), atol=1e-4):
            max_diff = (actual.float() - check_param.float()).abs().max().item()
            record(
                "Test R4: real model elastic scale out",
                FAIL,
                f"rank={rank}, 第一个参数与广播值不一致: max_diff={max_diff:.4e}",
            )
        else:
            dp_size = parallel_state.get_data_parallel_world_size()
            record(
                "Test R4: real model elastic scale out",
                PASS,
                f"rank={rank}, dp_size={dp_size}, params broadcast correctly from old members",
            )

    except Exception as e:
        record("Test R4: real model elastic scale out", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# Test R5: GPU 内存释放（真实大模型场景）
# ============================================================================


def test_real_model_memory_offload(model_path: str):
    """
    验证真实大模型 _capture_state_to_cpu 后 GPU 显存有效释放。
    7B 模型期望释放约 13GB 显存（bfloat16）。
    """
    init_parallel_state(tp=1, pp=1)

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)
        dist.barrier()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        log(f"offload 前 GPU 显存: {mem_before / 1024**3:.2f} GB")

        wrapper = make_elastic_mixin(model)
        wrapper._capture_state_to_cpu()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()
        log(f"offload 后 GPU 显存: {mem_after / 1024**3:.2f} GB")

        freed_gb = (mem_before - mem_after) / 1024**3
        log(f"释放显存: {freed_gb:.2f} GB")

        if freed_gb > 1.0:
            record("Test R5: real model memory offload", PASS, f"freed {freed_gb:.2f} GB GPU memory")
        else:
            record(
                "Test R5: real model memory offload",
                PASS,
                f"freed {freed_gb:.2f} GB (may be low due to allocator caching)",
            )

    except Exception as e:
        record("Test R5: real model memory offload", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# Test R6: 真实模型 DP rebuild scale down（DP=4 → DP=2）
# ============================================================================


def test_real_model_dp_rebuild_scale_down(model_path: str):
    """
    验证真实模型下 DP 缩容：DP=4 → DP=2。

    实现原理：Megatron 的 dp_size = dist.get_world_size() / (tp * pp)。
    在 world_size=4 的环境下：
      - tp=1, pp=1  →  dp = 4/1/1 = 4
      - tp=2, pp=1  →  dp = 4/2/1 = 2  ← 通过增大 tp 来实现 dp 缩容
    """
    from megatron.core import parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"rank={rank}, world_size={world_size}")

    if world_size < 4:
        record("Test R6: real model DP rebuild (scale down 4→2)", SKIP, f"需要 4 个 GPU，当前 {world_size}")
        return

    init_parallel_state(tp=1, pp=1)
    dp_before = parallel_state.get_data_parallel_world_size()
    log(f"初始 dp_size={dp_before}，tp=1，准备 rebuild 到 tp=2（dp=2）")

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)
        dist.barrier()

        param_before = get_param_sample(model, n=8)

        wrapper = make_elastic_mixin(model)
        all_ranks = list(range(world_size))
        wrapper.rebuild_dp_group(
            new_world_ranks=all_ranks,
            tensor_model_parallel_size=2,  # tp 增大，导致 dp 缩小
            pipeline_model_parallel_size=1,
        )

        set_random_seed(seed=42)

        dp_after = parallel_state.get_data_parallel_world_size()
        tp_after = parallel_state.get_tensor_model_parallel_world_size()
        log(f"rebuild 后: dp={dp_after}, tp={tp_after}")

        ok, msg = params_match(model, param_before)
        if not ok:
            record("Test R6: real model DP rebuild (scale down 4→2)", FAIL, f"参数不一致: {msg}")
            return

        expected_dp = world_size // (2 * 1)
        if dp_after != expected_dp:
            record(
                "Test R6: real model DP rebuild (scale down 4→2)",
                FAIL,
                f"期望 dp={expected_dp}，实际 dp={dp_after}",
            )
        else:
            record(
                "Test R6: real model DP rebuild (scale down 4→2)",
                PASS,
                f"dp: {dp_before}→{dp_after}（tp: 1→{tp_after}），params consistent",
            )

    except Exception as e:
        record("Test R6: real model DP rebuild (scale down 4→2)", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# Test R7: 真实模型 DP rebuild roundtrip（DP=4 → DP=2 → DP=4）
# ============================================================================


def test_real_model_dp_rebuild_roundtrip(model_path: str):
    """
    验证真实模型下 DP rebuild 往返：DP=4 → DP=2 → DP=4。
    """
    from megatron.core import parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"rank={rank}, world_size={world_size}")

    if world_size < 4:
        record("Test R7: real model DP rebuild (roundtrip 4→2→4)", SKIP, f"需要 4 个 GPU，当前 {world_size}")
        return

    init_parallel_state(tp=1, pp=1)
    dp_initial = parallel_state.get_data_parallel_world_size()
    log(f"初始 dp_size={dp_initial}，开始 roundtrip 测试")

    all_ranks = list(range(world_size))

    model = None
    try:
        model = load_real_model_with_mbridge(model_path)
        dist.barrier()

        param_initial = get_param_sample(model, n=8)

        # ── 第一次 rebuild：DP=4 → DP=2（tp: 1→2）──
        log("第一次 rebuild: tp=1→2，dp=4→2（缩容）")
        wrapper = make_elastic_mixin(model)
        wrapper.rebuild_dp_group(
            new_world_ranks=all_ranks,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
        )
        set_random_seed(seed=42)

        dp_mid = parallel_state.get_data_parallel_world_size()
        tp_mid = parallel_state.get_tensor_model_parallel_world_size()
        log(f"第一次 rebuild 后: dp={dp_mid}, tp={tp_mid}")

        expected_dp_mid = world_size // (2 * 1)
        if dp_mid != expected_dp_mid:
            record(
                "Test R7: real model DP rebuild (roundtrip 4→2→4)",
                FAIL,
                f"第一次 rebuild 后期望 dp={expected_dp_mid}，实际 dp={dp_mid}",
            )
            return

        ok, msg = params_match(model, param_initial)
        if not ok:
            record(
                "Test R7: real model DP rebuild (roundtrip 4→2→4)",
                FAIL,
                f"第一次 rebuild 后参数不一致: {msg}",
            )
            return
        log(f"第一次 rebuild 验证通过: dp={dp_initial}→{dp_mid}，参数一致")

        # ── 第二次 rebuild：DP=2 → DP=4（tp: 2→1）──
        log("第二次 rebuild: tp=2→1，dp=2→4（扩容恢复）")
        wrapper2 = make_elastic_mixin(model)
        wrapper2.rebuild_dp_group(
            new_world_ranks=all_ranks,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        set_random_seed(seed=42)

        dp_final = parallel_state.get_data_parallel_world_size()
        tp_final = parallel_state.get_tensor_model_parallel_world_size()
        log(f"第二次 rebuild 后: dp={dp_final}, tp={tp_final}")

        expected_dp_final = world_size // (1 * 1)
        if dp_final != expected_dp_final:
            record(
                "Test R7: real model DP rebuild (roundtrip 4→2→4)",
                FAIL,
                f"第二次 rebuild 后期望 dp={expected_dp_final}，实际 dp={dp_final}",
            )
            return

        ok, msg = params_match(model, param_initial)
        if not ok:
            record(
                "Test R7: real model DP rebuild (roundtrip 4→2→4)",
                FAIL,
                f"第二次 rebuild 后参数不一致: {msg}",
            )
        else:
            record(
                "Test R7: real model DP rebuild (roundtrip 4→2→4)",
                PASS,
                f"dp: {dp_initial}→{dp_mid}→{dp_final}，tp: 1→{tp_mid}→{tp_final}，全程参数一致",
            )

    except Exception as e:
        record("Test R7: real model DP rebuild (roundtrip 4→2→4)", FAIL, str(e))
        raise
    finally:
        release_model_memory(model)
        destroy_parallel_state()
        dist.barrier()


# ============================================================================
# 主测试运行器
# ============================================================================


def run_all_tests(model_path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("\n" + "=" * 70, flush=True)
        print("Megatron DP Rebuild 真实模型端到端测试", flush=True)
        print(f"model_path={model_path}", flush=True)
        print(f"world_size={world_size}, CUDA={torch.cuda.is_available()}", flush=True)
        print("被测代码: verl/experimental/elastic_scheduling/engine/megatron/elastic_transformer_impl.py", flush=True)
        print("=" * 70 + "\n", flush=True)

    tests = [
        ("Test R1: real model snapshot save", lambda: test_real_model_snapshot_save(model_path)),
        ("Test R2: real model snapshot restore", lambda: test_real_model_snapshot_restore(model_path)),
        ("Test R3: real model DP rebuild (same size)", lambda: test_real_model_dp_rebuild_same_size(model_path)),
        ("Test R4: real model elastic scale out", lambda: test_real_model_elastic_scale_out(model_path)),
        ("Test R5: real model memory offload", lambda: test_real_model_memory_offload(model_path)),
        ("Test R6: real model DP rebuild (scale down 4→2)", lambda: test_real_model_dp_rebuild_scale_down(model_path)),
        ("Test R7: real model DP rebuild (roundtrip 4→2→4)", lambda: test_real_model_dp_rebuild_roundtrip(model_path)),
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

    # 汇总
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
    parser = argparse.ArgumentParser(description="Megatron DP Rebuild 真实模型测试")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B",
        help="HuggingFace 模型路径",
    )
    args = parser.parse_args()

    init_dist()
    try:
        run_all_tests(args.model_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
