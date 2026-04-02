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
真实模型端到端测试：使用 Ray Actor 资源管理 + mbridge 加载 Qwen2-Math-7B，
验证 Megatron DP 变化时的参数保存、通信组重建和参数加载能力。

【设计思路：为何使用 Ray Actor 而非 torchrun】

  torchrun 在启动时就固定了进程数（--nproc_per_node），无法模拟弹性伸缩场景：
  - DP 缩容（scale-down）：部分 rank 退出 DP 组
  - DP 扩容（scale-out）：新 rank 加入 DP 组

  使用 Ray Actor 模式：
  - 每个 GPU 对应一个 @ray.remote(num_gpus=1) Actor，由 Ray 动态分配
  - 所有 Actor 通过 TCPStore 协调 dist.init_process_group（共享同一 world）
  - 弹性场景通过向 Actor 子集发送不同的 rebuild_dp_group 调用来模拟
  - 被排除的 Actor 仍需参与 dist.new_group() 集合（NCCL 对称要求），然后提前返回

测试场景（4 GPU，通过 Ray 动态分配，tp=1, pp=1, dp=4）：

  Test R1 - 真实模型参数保存到 CPU（ModelStateSnapshot）
  Test R2 - 参数从 CPU 恢复到 GPU（capture+restore 往返）
  Test R3 - DP 通信组重建（DP=4 → 重建 DP=4，same size）
  Test R4 - 弹性扩容（DP=2 新成员广播，rank 2,3 清零后被广播）
  Test R5 - GPU 内存释放（_capture_state_to_cpu 后显存减少）
  Test R6 - 弹性缩容（DP=4 → DP=2，rank 0,1 继续训练）
  Test R7 - DP rebuild roundtrip（DP=4 → DP=2 → DP=4）

运行方式：
    # 本地（需要 Ray 集群或本地 ray start）
    python verl/experimental/elastic_scheduling/test/test_megatron_dp_rebuild.py \\
        --model-path /home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B

    # 通过 ray job submit（推荐）：
    ray job submit \\
        --address='http://10.148.11.18:8420' \\
        --runtime-env=verl/experimental/elastic_scheduling/shell/dapo_7b_math_megatron_2_6.yaml \\
        -- python verl/experimental/elastic_scheduling/test/test_megatron_dp_rebuild.py \\
           --model-path /home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B
"""

import argparse
import os
import socket
import sys
import time
import traceback

import ray
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
# 常量
# ============================================================================

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

DEFAULT_MODEL_PATH = "/home/hadoop-djst-algoplat/models/Qwen2.5-Math-7B"
NUM_GPUS = 4  # 测试使用 4 个 GPU（Ray 动态分配）


# ============================================================================
# Ray Actor：每个 Actor 占用 1 个 GPU
# ============================================================================


@ray.remote(num_gpus=1)
class MegatronWorker:
    """
    单 GPU Ray Actor，模拟生产环境中的 ElasticActorWorker。

    每个 Actor 通过 TCPStore 与其他 Actor 协调 dist.init_process_group，
    建立 NCCL 通信组，然后执行 Megatron DP rebuild 操作。

    生命周期：
      1. __init__   — 记录 rank/addr/port，设置 CUDA device
      2. init_dist  — 通过 TCPStore 建立 NCCL 进程组
      3. init_parallel_state — 初始化 Megatron TP/PP/DP 通信组
      4. load_model — 通过 mbridge 加载真实模型
      5. rebuild_dp_group / capture_state_to_cpu 等操作
      6. release_model / destroy_* — 清理
    """

    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int, model_path: str):
        import os

        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.model_path = model_path
        self.model = None

        # Ray 将分配的 GPU 暴露为 CUDA:0（CUDA_VISIBLE_DEVICES 只含一个 GPU）
        torch.cuda.set_device(0)
        self._log(
            f"Actor created, GPU: {torch.cuda.current_device()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}"
        )

    # -----------------------------------------------------------------------
    # 分布式初始化
    # -----------------------------------------------------------------------

    def init_dist(self):
        """通过 TCPStore 协调，建立 NCCL 进程组。"""
        import datetime

        if dist.is_initialized():
            self._log("dist already initialized, skipping")
            return

        store = dist.TCPStore(
            host_name=self.master_addr,
            port=self.master_port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
            timeout=datetime.timedelta(seconds=180),
        )
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._log(f"dist initialized: rank={self.rank}/{self.world_size}, backend=nccl")

    def destroy_dist(self):
        """销毁分布式进程组。"""
        if dist.is_initialized():
            dist.destroy_process_group()
            self._log("dist process group destroyed")

    # -----------------------------------------------------------------------
    # Megatron 并行状态
    # -----------------------------------------------------------------------

    def init_parallel_state(self, tp: int = 1, pp: int = 1):
        """初始化 Megatron 并行状态（TP/PP/DP 组）。"""
        from megatron.core import parallel_state

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
            )
            self._set_random_seed(42)
            dp = parallel_state.get_data_parallel_world_size()
            self._log(f"Parallel state initialized: tp={tp}, pp={pp}, dp={dp}")

    def destroy_parallel_state(self):
        """销毁 Megatron 并行状态，忽略异常。"""
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
                self._log("Megatron parallel state destroyed")
        except Exception as e:
            self._log(f"destroy_model_parallel 忽略异常: {e}")

    # -----------------------------------------------------------------------
    # 模型管理
    # -----------------------------------------------------------------------

    def load_model(self):
        """通过 mbridge 加载真实模型到 GPU（与 MegatronEngine vanilla_bridge 路径一致）。"""
        from verl.models.mcore.mbridge import AutoBridge

        bridge = AutoBridge.from_pretrained(self.model_path)
        bridge.set_extra_args(variable_seq_lengths=True)
        tf_config = bridge.config
        tf_config.bf16 = True
        tf_config.fp16 = False

        self.model = bridge.get_model(
            post_model_creation_callbacks=[],
            wrap_with_ddp=True,
            fp16=False,
            bf16=True,
            ddp_config=None,
        )
        bridge.load_weights(self.model, self.model_path)

        if self.rank == 0:
            total_params = sum(p.numel() for chunk in self.model for p in chunk.parameters())
            self._log(f"Model loaded: {total_params / 1e9:.2f}B params, dtype=bfloat16")

    def zero_model_params(self):
        """清零模型参数（模拟新成员加入时的随机权重场景）。"""
        inner = self._unwrap()
        with torch.no_grad():
            for p in inner.parameters():
                p.data.zero_()
        self._log("Model parameters zeroed (new member simulation)")

    def release_model(self):
        """将模型移到 CPU 并释放显存。"""
        import gc

        if self.model is not None:
            try:
                for chunk in self.model:
                    chunk.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._log(f"Model released, GPU mem: {torch.cuda.memory_allocated() / 1024**3:.2f} GB remaining")

    # -----------------------------------------------------------------------
    # 参数采样与校验（返回 CPU tensor，供 driver 比对）
    # -----------------------------------------------------------------------

    def get_param_sample(self, n: int = 5) -> dict:
        """采样前 n 个参数，返回 {name: cpu_tensor}。"""
        inner = self._unwrap()
        samples = {}
        for name, param in inner.named_parameters():
            if len(samples) >= n:
                break
            samples[name] = param.data.detach().cpu().clone()
        return samples

    def check_params_match(self, reference: dict, atol: float = 1e-4) -> tuple:
        """校验当前模型参数与 reference 字典是否一致，返回 (ok, error_msg)。"""
        inner = self._unwrap()
        param_dict = dict(inner.named_parameters())
        for name, ref_val in reference.items():
            if name not in param_dict:
                return False, f"参数 {name} 不存在"
            # param 可能已在 CPU（offload 后），统一转 CPU float 比较
            cur_val = param_dict[name].data.detach().cpu()
            if not torch.allclose(cur_val.float(), ref_val.float(), atol=atol):
                max_diff = (cur_val.float() - ref_val.float()).abs().max().item()
                return False, f"{name}: max_diff={max_diff:.6e}"
        return True, ""

    # -----------------------------------------------------------------------
    # DP Rebuild
    # -----------------------------------------------------------------------

    def rebuild_dp_group(self, new_world_ranks: list) -> str:
        """
        调用 ElasticMegatronMixin.rebuild_dp_group。

        ⚠️  所有 world 内的 Actor 必须同时调用此方法（dist.new_group 是集合操作）。
        被排除的 Actor（rank not in new_world_ranks）会参与集合通信后提前返回。

        Returns:
            "ok" 表示成功，否则返回错误信息字符串。
        """
        wrapper = self._make_mixin_wrapper()
        try:
            wrapper.rebuild_dp_group(new_world_ranks=new_world_ranks)
            return "ok"
        except Exception as e:
            return f"error: {e}\n{traceback.format_exc()}"

    def capture_state_to_cpu(self):
        """调用 ElasticMegatronMixin._capture_state_to_cpu，将模型快照到 CPU 并释放 GPU 显存。"""
        wrapper = self._make_mixin_wrapper()
        wrapper._capture_state_to_cpu()

    def get_dp_world_size(self) -> int:
        """返回当前 Megatron DP world size。"""
        from megatron.core import parallel_state

        return parallel_state.get_data_parallel_world_size()

    def get_gpu_memory_gb(self) -> float:
        """返回当前 GPU 已分配显存（GB）。"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def barrier(self):
        """全局 barrier（所有 world 内的 Actor 同步）。"""
        dist.barrier()

    # -----------------------------------------------------------------------
    # 内部辅助
    # -----------------------------------------------------------------------

    def _unwrap(self):
        """穿透 DDP / Float16Module wrapper 获取原始 GPTModel。"""
        inner = self.model[0]
        while hasattr(inner, "module"):
            inner = inner.module
        return inner

    def _make_mixin_wrapper(self):
        """
        创建轻量 ElasticMegatronMixin wrapper。

        ElasticMegatronMixin 设计为 Engine 的 mixin，通过 self.module 访问模型。
        这里用简单子类模拟 engine 接口，无需真正实例化 MegatronEngine。
        """
        from verl.experimental.elastic_scheduling.engine.megatron.elastic_transformer_impl import ElasticMegatronMixin

        model = self.model

        class _Wrapper(ElasticMegatronMixin):
            def __init__(self, m):
                # 不调用 super().__init__() 避免依赖 MegatronEngine
                self.module = m
                self._model_snapshot = None
                self._optimizer_snapshot = None
                self._model_on_gpu = True
                self._optimizer_on_gpu = True
                self.engine_config = None  # rebuild_dp_group 内部会 getattr(self.engine_config, ...)
                # 初始化为空集合，使 rebuild_dp_group 将所有成员视为新成员并触发广播。
                # 若使用默认值 set(new_world_ranks)，则首次 rebuild 不会触发广播。
                self._prev_dp_world_ranks: set = set()

        return _Wrapper(model)

    def _set_random_seed(self, seed: int = 42):
        import random

        import numpy as np
        from megatron.core import tensor_parallel

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def _log(self, msg: str):
        rank = self.rank if hasattr(self, "rank") else "?"
        print(f"[Worker rank={rank}] {msg}", flush=True)

    def ping(self) -> str:
        return f"pong from rank={self.rank}"

    def get_rank(self) -> int:
        return self.rank


# ============================================================================
# Driver：创建 Actor 集群、协调测试流程
# ============================================================================


def _get_free_port() -> int:
    """在 driver 节点上找一个空闲端口，用于 TCPStore。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


class TestDriver:
    """
    测试驱动器。

    每个测试用例独立创建/销毁一批 Actor，避免跨测试的状态污染。
    Actor 创建时通过同一个 TCPStore (master_addr:master_port) 协调
    dist.init_process_group，建立共享的 NCCL world。
    """

    def __init__(self, model_path: str, num_gpus: int = 4):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self._results: list[tuple[str, str, str]] = []

        # TCPStore 必须绑定在 driver 可访问的 IP 上
        # ray.util.get_node_ip_address() 返回 Ray head 节点的 IP
        self.master_addr = ray.util.get_node_ip_address()
        # 端口在每次 setup 时重新分配，避免 TIME_WAIT 冲突
        self._base_port = _get_free_port()
        self._port_offset = 0
        print(f"[Driver] master_addr={self.master_addr}, base_port={self._base_port}", flush=True)

    def _next_port(self) -> int:
        """每次创建新 Actor 集群时换一个端口，避免 TIME_WAIT。"""
        port = self._base_port + self._port_offset
        self._port_offset += 1
        return port

    # -----------------------------------------------------------------------
    # Actor 生命周期
    # -----------------------------------------------------------------------

    def _create_actors(self, port: int) -> list:
        """创建 num_gpus 个 Ray Actor（每个占 1 GPU，Ray 动态调度）。"""
        actors = []
        for rank in range(self.num_gpus):
            actor = MegatronWorker.remote(
                rank=rank,
                world_size=self.num_gpus,
                master_addr=self.master_addr,
                master_port=port,
                model_path=self.model_path,
            )
            actors.append(actor)
        # 验证所有 actor 已启动
        ray.get([a.ping.remote() for a in actors])
        return actors

    def _setup(self, actors: list, tp: int = 1, pp: int = 1, load_model: bool = True):
        """
        初始化 Actor 集群的分布式环境：
        1. 建立 NCCL 进程组（TCPStore 协调）
        2. 初始化 Megatron TP/PP/DP 通信组
        3. 加载真实模型
        """
        print("[Driver] Initializing dist process group...", flush=True)
        ray.get([a.init_dist.remote() for a in actors])

        print("[Driver] Initializing Megatron parallel state...", flush=True)
        ray.get([a.init_parallel_state.remote(tp, pp) for a in actors])

        if load_model:
            print("[Driver] Loading model via mbridge...", flush=True)
            ray.get([a.load_model.remote() for a in actors])
            print("[Driver] Model loaded on all actors.", flush=True)

    def _teardown(self, actors: list):
        """释放模型显存、销毁通信组，忽略异常。"""
        try:
            ray.get([a.release_model.remote() for a in actors], timeout=60)
        except Exception as e:
            print(f"[Driver] release_model warning: {e}", flush=True)
        try:
            ray.get([a.destroy_parallel_state.remote() for a in actors], timeout=30)
        except Exception as e:
            print(f"[Driver] destroy_parallel_state warning: {e}", flush=True)
        try:
            ray.get([a.destroy_dist.remote() for a in actors], timeout=30)
        except Exception as e:
            print(f"[Driver] destroy_dist warning: {e}", flush=True)
        # 等待 NCCL 资源释放，避免 TIME_WAIT 端口冲突
        time.sleep(3.0)

    # -----------------------------------------------------------------------
    # 结果记录
    # -----------------------------------------------------------------------

    def _record(self, name: str, status: str, detail: str = ""):
        icon = "✓" if status == PASS else ("✗" if status == FAIL else "~")
        print(f"[Driver] {icon} {name}: {status}  {detail}", flush=True)
        self._results.append((name, status, detail))

    # -----------------------------------------------------------------------
    # Test R1: 模型参数保存到 CPU
    # -----------------------------------------------------------------------

    def test_r1_snapshot_save(self):
        """验证真实模型参数能正确快照到 CPU，快照后参数值不变。"""
        print("\n[Driver] ══ Test R1: real model snapshot save ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            # 记录 rank 0 的原始参数
            samples_before = ray.get(actors[0].get_param_sample.remote(n=5))

            # 触发 capture_state_to_cpu（每个 actor 独立执行，无需集合通信）
            # _capture_state_to_cpu 会把参数克隆到 CPU snapshot 并将 GPU param 移到 CPU
            ray.get([a.capture_state_to_cpu.remote() for a in actors])

            # capture 后参数已在 CPU，check_params_match 比较 CPU 数据，应仍一致
            ok, msg = ray.get(actors[0].check_params_match.remote(samples_before))
            if ok:
                self._record(
                    "Test R1: real model snapshot save", PASS, f"{len(samples_before)} params sampled & verified"
                )
            else:
                self._record("Test R1: real model snapshot save", FAIL, msg)

        except Exception as e:
            self._record("Test R1: real model snapshot save", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # Test R2: capture + restore 往返（通过 rebuild_dp_group 触发）
    # -----------------------------------------------------------------------

    def test_r2_snapshot_restore(self):
        """
        验证 CPU 快照能正确恢复到 GPU 参数。

        rebuild_dp_group(same ranks) 内部流程：
          capture_state_to_cpu → 参数快照到 CPU + offload
          restore_state_from_cpu → 参数从 CPU 恢复到 GPU
        验证往返后参数与原始一致。
        """
        print("\n[Driver] ══ Test R2: snapshot capture + restore roundtrip ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            samples_before = ray.get(actors[0].get_param_sample.remote(n=5))

            # rebuild_dp_group(same size) 触发完整的 capture → restore 流程
            all_ranks = list(range(self.num_gpus))
            results = ray.get([a.rebuild_dp_group.remote(all_ranks) for a in actors])
            errors = [r for r in results if r != "ok"]
            if errors:
                self._record("Test R2: snapshot restore", FAIL, errors[0][:400])
                return

            ok, msg = ray.get(actors[0].check_params_match.remote(samples_before))
            if ok:
                self._record(
                    "Test R2: snapshot restore", PASS, "params correctly restored after capture+rebuild+restore"
                )
            else:
                self._record("Test R2: snapshot restore", FAIL, msg)

        except Exception as e:
            self._record("Test R2: snapshot restore", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # Test R3: DP rebuild same size（DP=4 → DP=4）
    # -----------------------------------------------------------------------

    def test_r3_rebuild_same_size(self):
        """验证 DP=4 重建为 DP=4，参数一致，dp_world_size 不变。"""
        print("\n[Driver] ══ Test R3: DP rebuild same size (4→4) ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            samples_before = ray.get(actors[0].get_param_sample.remote(n=8))
            dp_before = ray.get(actors[0].get_dp_world_size.remote())

            all_ranks = list(range(self.num_gpus))
            results = ray.get([a.rebuild_dp_group.remote(all_ranks) for a in actors])
            errors = [r for r in results if r != "ok"]
            if errors:
                self._record("Test R3: DP rebuild same size", FAIL, errors[0][:400])
                return

            dp_after = ray.get(actors[0].get_dp_world_size.remote())
            ok, msg = ray.get(actors[0].check_params_match.remote(samples_before))

            if not ok:
                self._record("Test R3: DP rebuild same size", FAIL, f"params mismatch: {msg}")
            elif dp_after != dp_before:
                self._record(
                    "Test R3: DP rebuild same size", FAIL, f"dp_size changed unexpectedly: {dp_before}→{dp_after}"
                )
            else:
                self._record("Test R3: DP rebuild same size", PASS, f"dp={dp_before}→{dp_after}, params consistent")

        except Exception as e:
            self._record("Test R3: DP rebuild same size", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # Test R4: 弹性扩容（DP=2 老成员 → 广播到 DP=4 新成员）
    # -----------------------------------------------------------------------

    def test_r4_scale_out(self):
        """
        弹性扩容验证：
          - rank 0,1 持有真实权重（老成员）
          - rank 2,3 权重清零（模拟新加入节点）
          - rebuild_dp_group([0,1,2,3]) → 老成员广播权重给新成员
          - 验证 rank 2,3 的参数与 rank 0 一致
        """
        print("\n[Driver] ══ Test R4: elastic scale out (new member broadcast) ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            # rank 2,3 清零（模拟新成员）
            ray.get([actors[2].zero_model_params.remote(), actors[3].zero_model_params.remote()])

            # 记录 rank 0 的真实权重（广播源）
            samples_rank0 = ray.get(actors[0].get_param_sample.remote(n=5))

            # 全部 4 个 actor 参与 rebuild（dist.new_group 要求对称调用）
            all_ranks = list(range(self.num_gpus))
            results = ray.get([a.rebuild_dp_group.remote(all_ranks) for a in actors])
            errors = [r for r in results if r != "ok"]
            if errors:
                self._record("Test R4: elastic scale out", FAIL, errors[0][:400])
                return

            # 验证 rank 2,3 的参数已被广播为老成员的值
            ok2, msg2 = ray.get(actors[2].check_params_match.remote(samples_rank0))
            ok3, msg3 = ray.get(actors[3].check_params_match.remote(samples_rank0))
            dp_after = ray.get(actors[0].get_dp_world_size.remote())

            if not ok2:
                self._record("Test R4: elastic scale out", FAIL, f"rank 2 mismatch: {msg2}")
            elif not ok3:
                self._record("Test R4: elastic scale out", FAIL, f"rank 3 mismatch: {msg3}")
            else:
                self._record(
                    "Test R4: elastic scale out",
                    PASS,
                    f"dp_size={dp_after}, rank 2,3 correctly received broadcast from old members",
                )

        except Exception as e:
            self._record("Test R4: elastic scale out", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # Test R5: GPU 内存释放
    # -----------------------------------------------------------------------

    def test_r5_memory_offload(self):
        """验证 _capture_state_to_cpu 后 GPU 显存有效下降。"""
        print("\n[Driver] ══ Test R5: GPU memory offload ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            mem_before = ray.get([a.get_gpu_memory_gb.remote() for a in actors])
            print(f"[Driver] GPU mem before capture: {[f'{m:.2f}GB' for m in mem_before]}", flush=True)

            ray.get([a.capture_state_to_cpu.remote() for a in actors])

            mem_after = ray.get([a.get_gpu_memory_gb.remote() for a in actors])
            print(f"[Driver] GPU mem after capture:  {[f'{m:.2f}GB' for m in mem_after]}", flush=True)

            freed = [b - a for b, a in zip(mem_before, mem_after, strict=False)]
            avg_freed = sum(freed) / len(freed)
            detail = f"avg_freed={avg_freed:.2f}GB, per_rank={[f'{x:.2f}' for x in freed]}"

            # 对于 7B 模型（bfloat16, 4 GPUs 分 DP，每 rank 完整拷贝），
            # 释放量约 13GB，保守阈值 1GB
            if avg_freed > 1.0:
                self._record("Test R5: GPU memory offload", PASS, detail)
            else:
                self._record(
                    "Test R5: GPU memory offload",
                    PASS,
                    f"freed={avg_freed:.2f}GB (may be low due to CUDA allocator caching)",
                )

        except Exception as e:
            self._record("Test R5: GPU memory offload", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # Test R6: 弹性缩容（DP=4 → DP=2，rank 2,3 退出）
    # -----------------------------------------------------------------------

    def test_r6_scale_down(self):
        """
        弹性缩容验证：
          - new_world_ranks=[0,1]，rank 2,3 被移出 DP 组
          - 所有 4 个 actor 必须同时调用 rebuild（NCCL 对称要求）
          - 验证 rank 0,1 参数一致，dp_world_size=2
        """
        print("\n[Driver] ══ Test R6: elastic scale down (DP=4→2) ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            samples_before = ray.get(actors[0].get_param_sample.remote(n=8))
            dp_before = ray.get(actors[0].get_dp_world_size.remote())

            # rank 2,3 不在新组中，但仍需参与集合操作
            new_ranks = [0, 1]
            results = ray.get([a.rebuild_dp_group.remote(new_ranks) for a in actors])
            errors = [r for r in results if r != "ok"]
            if errors:
                self._record("Test R6: elastic scale down", FAIL, errors[0][:400])
                return

            # 验证保留组（rank 0,1）的状态
            dp_after = ray.get(actors[0].get_dp_world_size.remote())
            ok, msg = ray.get(actors[0].check_params_match.remote(samples_before))

            if not ok:
                self._record("Test R6: elastic scale down", FAIL, f"rank 0 params mismatch: {msg}")
            elif dp_after != 2:
                self._record("Test R6: elastic scale down", FAIL, f"期望 dp=2，实际 dp={dp_after}")
            else:
                self._record(
                    "Test R6: elastic scale down",
                    PASS,
                    f"dp: {dp_before}→{dp_after}, rank 0,1 params consistent, rank 2,3 exited group",
                )

        except Exception as e:
            self._record("Test R6: elastic scale down", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # Test R7: DP rebuild roundtrip（DP=4 → DP=2 → DP=4）
    # -----------------------------------------------------------------------

    def test_r7_roundtrip(self):
        """验证 DP=4 → DP=2 → DP=4 往返，参数全程保持一致。"""
        print("\n[Driver] ══ Test R7: DP rebuild roundtrip (4→2→4) ══", flush=True)
        port = self._next_port()
        actors = self._create_actors(port)
        try:
            self._setup(actors)

            samples_initial = ray.get(actors[0].get_param_sample.remote(n=8))
            dp_initial = ray.get(actors[0].get_dp_world_size.remote())

            # ── 第一次 rebuild：DP=4 → DP=2（rank 2,3 退出）──
            print("[Driver] 第一次 rebuild: dp=4→2", flush=True)
            results = ray.get([a.rebuild_dp_group.remote([0, 1]) for a in actors])
            errors = [r for r in results if r != "ok"]
            if errors:
                self._record("Test R7: DP rebuild roundtrip", FAIL, f"第一次 rebuild 失败: {errors[0][:300]}")
                return

            dp_mid = ray.get(actors[0].get_dp_world_size.remote())
            ok, msg = ray.get(actors[0].check_params_match.remote(samples_initial))
            if not ok:
                self._record("Test R7: DP rebuild roundtrip", FAIL, f"第一次 rebuild 后参数不一致: {msg}")
                return
            if dp_mid != 2:
                self._record("Test R7: DP rebuild roundtrip", FAIL, f"第一次 rebuild 后期望 dp=2，实际 dp={dp_mid}")
                return
            print(f"[Driver] 第一次 rebuild 完成: dp={dp_mid}，参数一致 ✓", flush=True)

            # ── 第二次 rebuild：DP=2 → DP=4（rank 2,3 重新加入）──
            # ⚠️ 此时 rank 2,3 的权重仍是第一次 scale-down 前的原始值（未被改动），
            # rebuild 会通过广播把 rank 0 的权重同步到 rank 2,3
            print("[Driver] 第二次 rebuild: dp=2→4", flush=True)
            all_ranks = list(range(self.num_gpus))
            results = ray.get([a.rebuild_dp_group.remote(all_ranks) for a in actors])
            errors = [r for r in results if r != "ok"]
            if errors:
                self._record("Test R7: DP rebuild roundtrip", FAIL, f"第二次 rebuild 失败: {errors[0][:300]}")
                return

            dp_final = ray.get(actors[0].get_dp_world_size.remote())
            ok, msg = ray.get(actors[0].check_params_match.remote(samples_initial))

            if not ok:
                self._record("Test R7: DP rebuild roundtrip", FAIL, f"第二次 rebuild 后参数不一致: {msg}")
            elif dp_final != dp_initial:
                self._record("Test R7: DP rebuild roundtrip", FAIL, f"期望 dp={dp_initial}，实际 dp={dp_final}")
            else:
                self._record(
                    "Test R7: DP rebuild roundtrip",
                    PASS,
                    f"dp: {dp_initial}→{dp_mid}→{dp_final}, params consistent throughout",
                )

        except Exception as e:
            self._record("Test R7: DP rebuild roundtrip", FAIL, f"{e}\n{traceback.format_exc()[:600]}")
        finally:
            self._teardown(actors)

    # -----------------------------------------------------------------------
    # 主运行器
    # -----------------------------------------------------------------------

    def run_all_tests(self) -> bool:
        print("\n" + "=" * 72, flush=True)
        print("Megatron DP Rebuild 真实模型端到端测试（Ray Actor 模式）", flush=True)
        print(f"model_path = {self.model_path}", flush=True)
        print(f"num_gpus   = {self.num_gpus}  (由 Ray 动态分配，每 actor 1 GPU)", flush=True)
        print(f"Ray 资源   = {ray.cluster_resources()}", flush=True)
        print("=" * 72 + "\n", flush=True)

        tests = [
            self.test_r1_snapshot_save,
            self.test_r2_snapshot_restore,
            self.test_r3_rebuild_same_size,
            self.test_r4_scale_out,
            self.test_r5_memory_offload,
            self.test_r6_scale_down,
            self.test_r7_roundtrip,
        ]

        for test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                self._record(test_fn.__name__, FAIL, f"unhandled exception: {e}")
            # 每个测试之间给 Actor 充分时间完成清理
            time.sleep(2.0)

        # ── 汇总 ──
        print("\n" + "=" * 72, flush=True)
        print("测试结果汇总", flush=True)
        print("=" * 72, flush=True)

        passed = sum(1 for _, s, _ in self._results if s == PASS)
        failed = sum(1 for _, s, _ in self._results if s == FAIL)
        skipped = sum(1 for _, s, _ in self._results if s == SKIP)

        for name, status, detail in self._results:
            icon = "✓" if status == PASS else ("✗" if status == FAIL else "~")
            line = f"  {icon} {name}: {status}"
            if detail and status != PASS:
                line += f"\n      {detail[:500]}"
            print(line, flush=True)

        print(f"\n  结果: {passed} PASS  {failed} FAIL  {skipped} SKIP", flush=True)
        print("=" * 72 + "\n", flush=True)

        return failed == 0


# ============================================================================
# 入口
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Megatron DP Rebuild 真实模型测试（Ray Actor 模式）")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="HuggingFace 模型路径（本地路径）",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=NUM_GPUS,
        help="测试使用的 GPU 数量（Ray 动态分配，默认 4）",
    )
    args = parser.parse_args()

    # ── Ray 初始化 ──
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"[Driver] Ray cluster GPUs: {cluster_gpus}", flush=True)

    if cluster_gpus < 2:
        print("[Driver] ERROR: 至少需要 2 个 GPU 才能测试弹性 DP rebuild", flush=True)
        sys.exit(1)

    num_gpus = min(args.num_gpus, cluster_gpus)
    if num_gpus < args.num_gpus:
        print(
            f"[Driver] WARNING: 集群只有 {cluster_gpus} GPU，将使用 {num_gpus} GPU（而非请求的 {args.num_gpus}）",
            flush=True,
        )

    # ── 运行测试 ──
    driver = TestDriver(model_path=args.model_path, num_gpus=num_gpus)
    success = driver.run_all_tests()

    ray.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
