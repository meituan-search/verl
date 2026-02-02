# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
统一参数同步模块 (Unified Parameter Sync Module)

支持:
- 训练框架: FSDP (v1/v2), Megatron
- 推理引擎: vLLM, SGLang
- 硬件设备: GPU, NPU

设计原则:
1. 统一的接口 - 无论训练框架和推理引擎如何，接口保持一致
2. 可扩展性 - 易于添加新的训练框架或推理引擎支持
3. 高效同步 - 优化参数同步的性能
"""

import asyncio
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed
from ray.util.collective import collective

from verl.utils.device import (
    get_torch_device,
    is_npu_available,
)
from verl.utils.distributed import stateless_init_process_group

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ==================== 设备与后端相关 ====================

def get_inference_model(rollout) -> torch.nn.Module:
    """
    Get inference model from different rollout types.

    Args:
        rollout: rollout object (vLLM or SGLang)

    Returns:
        Model object for weight loading
    """
    inference_engine = rollout.inference_engine
    if hasattr(inference_engine, "llm_engine"):
        inference_model = (
            inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        )
    elif hasattr(inference_engine, "worker"):
        inference_model = inference_engine.worker.model_runner.model
    else:
        raise AttributeError(
            f"Unsupported inference_engine type: {type(inference_engine)}. "
            f"Expected LLM (with llm_engine attribute) or WorkerWrapperBase (with worker attribute)."
        )
    return inference_model


def broadcast_tensor(
    tensor: torch.Tensor,
    src_rank: int = 0,
    group_name: str = "actor_rollout",
    sync_group=None,
) -> None:
    """
    Broadcast tensor from source rank to all other ranks.

    Supports both GPU (NCCL) and NPU (HCCL) backends.

    Args:
        tensor: Tensor to broadcast (modified in-place)
        src_rank: Source rank
        group_name: Ray collective group name
        sync_group: PyNcclCommunicator or PyHcclCommunicator (for NPU)
    """
    if is_npu_available:
        # For NPU, use the sync_group communicator
        if sync_group is not None:
            sync_group.broadcast(tensor, src=src_rank, stream=get_torch_device().current_stream())
        else:
            raise ValueError("sync_group is required for NPU backend")
    else:
        # For GPU, use ray collective
        collective.broadcast(tensor, src_rank=src_rank, group_name=group_name)


# ==================== 权重信息相关 ====================

@dataclass
class WeightInfo:
    """权重信息数据类"""
    key: str  # 权重名称
    shape: Tuple[int, ...]  # 形状
    dtype: torch.dtype  # 数据类型

    def to_list(self) -> List:
        return [self.key, list(self.shape), self.dtype]


def build_weights_info(params: Dict[str, torch.Tensor]) -> List[WeightInfo]:
    """
    Build weight info list from state dict.

    Args:
        params: State dict from model

    Returns:
        List of WeightInfo objects
    """
    return [WeightInfo(key=key, shape=tuple(tensor.shape), dtype=tensor.dtype) for key, tensor in params.items()]


def serialize_weights_info(weights_info: List[WeightInfo]) -> List:
    """Serialize weight info to list format for transmission"""
    return [info.to_list() for info in weights_info]


def deserialize_weights_info(data: List) -> List[WeightInfo]:
    """Deserialize weight info from list format"""
    return [WeightInfo(key=item[0], shape=tuple(item[1]), dtype=item[2]) for item in data]


# ==================== 抽象基类 ====================

class ModelBackend(ABC):
    """模型后端抽象基类 (FSDP/Megatron)"""

    @abstractmethod
    def load_model_to_gpu(self) -> None:
        """Load model to GPU"""
        pass

    @abstractmethod
    def offload_model_to_cpu(self) -> None:
        """Offload model to CPU"""
        pass

    @abstractmethod
    def get_params_generator(
        self, model, model_config, weight_converter, tf_config, layer_name_mapping
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get parameter generator for weight sync"""
        pass

    @abstractmethod
    def get_weights_info(self, model) -> List[WeightInfo]:
        """Get weight information from model"""
        pass


class InferenceBackend(ABC):
    """推理后端抽象基类 (vLLM/SGLang)"""

    @abstractmethod
    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]) -> None:
        """Load weights into inference model"""
        pass

    @abstractmethod
    def resume_memory_occupation(self) -> None:
        """Resume memory occupation after weight sync"""
        pass


# ==================== FSDP 后端实现 ====================

class FSDPBackend:
    """FSDP 模型后端实现"""

    @staticmethod
    def load_model_to_gpu(model) -> None:
        from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu

        if fsdp_version(model) == 1:
            load_fsdp_model_to_gpu(model)
        else:
            load_fsdp_model_to_gpu(model)

    @staticmethod
    def offload_model_to_cpu(model) -> None:
        from verl.utils.fsdp_utils import fsdp_version, offload_fsdp_model_to_cpu

        if fsdp_version(model) == 1:
            offload_fsdp_model_to_cpu(model)
        else:
            offload_fsdp_model_to_cpu(model)

    @staticmethod
    def get_params(model) -> Dict[str, torch.Tensor]:
        """Get parameters from FSDP model state dict"""
        from verl.utils.model import convert_weight_keys

        params = model.state_dict()
        params = convert_weight_keys(params, getattr(model, "_fsdp_wrapped_module", model))
        return params

    @staticmethod
    def get_weights_info(model) -> List[WeightInfo]:
        """Get weight info from FSDP model"""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
        from verl.utils.fsdp_utils import fsdp_version

        if fsdp_version(model) == 1:
            FSDP.set_state_dict_type(
                model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        params = model.state_dict()
        return build_weights_info(params)

    @staticmethod
    def get_params_with_full_tensor(params: Dict[str, torch.Tensor]) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get parameters, handling full_tensor if available"""
        for key, tensor in params.items():
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            yield key, tensor

    @staticmethod
    def _create_empty_tensor(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Create empty tensor on current device"""
        return torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())


# ==================== Megatron 后端实现 ====================

class MegatronBackend:
    """Megatron 模型后端实现"""

    @staticmethod
    def load_model_to_gpu(model) -> None:
        from verl.utils.megatron_utils import load_megatron_model_to_gpu

        load_megatron_model_to_gpu(model)

    @staticmethod
    def offload_model_to_cpu(model) -> None:
        from verl.utils.megatron_utils import offload_megatron_model_to_cpu

        offload_megatron_model_to_cpu(model)

    @staticmethod
    def get_params_generator(
        model, model_config, weight_converter, tf_config, layer_name_mapping
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get parameter generator for Megatron model"""
        from verl.utils.megatron_utils import per_tensor_generator

        return per_tensor_generator(
            model,
            model_config,
            weight_converter,
            tf_config,
            layer_name_mapping,
        )

    @staticmethod
    def get_weights_info(model) -> List[WeightInfo]:
        """Get weight info from Megatron model"""
        from megatron.core import parallel_state as mpu

        from verl.utils.megatron_utils import unwrap_model

        params_info = []
        vpp_size = len(model)

        # Collect all weight info across pipeline stages
        for scan_vpp_idx in range(vpp_size):
            unwrapped_model = unwrap_model(model[scan_vpp_idx])
            for name, param in unwrapped_model.named_parameters():
                params_info.append(WeightInfo(key=name, shape=tuple(param.shape), dtype=param.dtype))

        # Gather info from all PP ranks
        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_size = torch.distributed.get_world_size(group=pp_group)

        all_params_info = [None] * pp_size
        torch.distributed.all_gather_object(all_params_info, params_info, group=pp_group)

        # Combine all info (take first non-empty, as all ranks should have same keys)
        combined_info = []
        for info_list in all_params_info:
            if info_list:
                combined_info = info_list
                break

        return combined_info


# ==================== vLLM 推理后端 ====================

class VLLMBackend:
    """vLLM 推理后端实现"""

    @staticmethod
    def patch_moe_model_weight_loader(inference_model) -> None:
        """Patch MoE model weight loader if needed"""
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(inference_model)

    @staticmethod
    def load_weights(inference_model, weights: List[Tuple[str, torch.Tensor]]) -> None:
        """Load weights into vLLM model"""
        inference_model.load_weights(weights)


# ==================== SGLang 推理后端 ====================

class SGLangBackend:
    """SGLang 推理后端实现"""

    @staticmethod
    async def update_weights(
        inference_engine,
        params_batch: List[Tuple[str, torch.Tensor]],
        device_mesh_key: str,
        device_mesh: Dict,
    ) -> None:
        """Async update weights in SGLang"""
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        await sgl_update_weights(
            engine=inference_engine,
            params_batch=params_batch,
            device_mesh_key=device_mesh_key,
            device_mesh=device_mesh,
        )

    @staticmethod
    async def flush_cache(inference_engine) -> None:
        """Flush cache after weight update"""
        await inference_engine.flush_cache()

    @staticmethod
    async def resume_memory_occupation(inference_engine, tags: List[str] = None) -> None:
        """Resume memory occupation after weight sync"""
        if tags is None:
            tags = ["kv_cache"]
        await inference_engine.resume_memory_occupation(tags=tags)


# ==================== 权重同步核心逻辑 ====================

@dataclass
class SyncMetrics:
    """同步指标记录"""
    bucket_size_mb: float = 0.0
    sync_time: float = 0.0
    cache_time: float = 0.0
    register_time: float = 0.0
    update_time: float = 0.0


class UnifiedWeightSynchronizer:
    """
    统一权重同步器

    整合了 FSDP/Megatron + vLLM/SGLang 的权重同步逻辑
    支持 GPU 和 NPU
    """

    # 动态分桶配置
    _bucket_size_mb: float = 1024.0
    _max_bucket_size_mb: float = 8192.0
    _min_bucket_size_mb: float = 512.0
    _sync_history: List[Tuple[float, float]] = []
    _max_history_size: int = 20

    def __init__(
        self,
        model_backend: str = "fsdp",
        inference_backend: str = "vllm",
        device: str = "gpu",
        engine=None,
    ):
        """
        Initialize the unified weight synchronizer.

        Args:
            model_backend: Training model backend ("fsdp" or "megatron")
            inference_backend: Inference engine backend ("vllm" or "sglang")
            device: Device type ("gpu" or "npu")
            engine: Worker engine instance with get_per_tensor_param() method
        """
        self.model_backend_type = model_backend
        self.inference_backend_type = inference_backend
        self.device = device
        self._engine = engine

        # 初始化后端
        self._init_backends()

        # 同步组
        self._sync_group: Optional[Any] = None
        self._sync_group_name: str = "actor_rollout"

        # 检查点引擎（可选）
        self._checkpoint_engine = None

        # 权重信息
        self._weights_info: Optional[List[WeightInfo]] = None

        # 背景事件循环（用于 SGLang 异步操作）
        self._bg_loop = None
        self._bg_thread = None

        if self.inference_backend_type == "sglang":
            self._init_background_loop()

    def _init_backends(self) -> None:
        """初始化后端"""
        # 模型后端
        if self.model_backend_type == "fsdp":
            self.model_backend = FSDPBackend()
        elif self.model_backend_type == "megatron":
            self.model_backend = MegatronBackend()
        else:
            raise ValueError(f"Unsupported model backend: {self.model_backend_type}")

        # 推理后端
        if self.inference_backend_type == "vllm":
            self.inference_backend = VLLMBackend()
        elif self.inference_backend_type == "sglang":
            self.inference_backend = SGLangBackend()
        else:
            raise ValueError(f"Unsupported inference backend: {self.inference_backend_type}")

    def _init_background_loop(self) -> None:
        """初始化背景事件循环（用于 SGLang）"""
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._start_background_loop,
            args=(self._bg_loop,),
            name="weight_sync_async_worker",
            daemon=True,
        )
        self._bg_thread.start()
        logger.info(f"[UnifiedWeightSynchronizer] Background thread started. PID: {os.getpid()}")

    def _start_background_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """启动背景事件循环"""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"[UnifiedWeightSynchronizer] Background loop crashed: {e}")

    def _run_async_safely(self, coro) -> Any:
        """安全地运行异步协程"""
        if not self._bg_thread.is_alive():
            raise RuntimeError("Background thread is not running!")

        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return future.result()

    def init_sync_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        """
        初始化权重同步组

        Args:
            master_address: Master node address
            master_port: Master node port
            rank_offset: Rank offset
            world_size: World size
        """
        rank = torch.distributed.get_rank() + rank_offset

        if self.device == "npu":
            # NPU 使用 stateless process group
            self._sync_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                get_torch_device().current_device(),
            )
        else:
            # GPU 使用 ray collective
            pass  # Collective group is created elsewhere

    def sync_weights(
        self,
        model,
        inference_model,
        is_actor: bool,
        is_rollout: bool,
        params: Optional[Dict[str, torch.Tensor]] = None,
        use_checkpoint_engine: bool = False,
    ) -> SyncMetrics:
        """
        同步权重

        Args:
            model: Training model (actor)
            inference_model: Inference model (rollout)
            is_actor: Whether this is actor
            is_rollout: Whether this is rollout
            params: Parameters from actor (for actor role)
            use_checkpoint_engine: Whether to use checkpoint engine for sync

        Returns:
            SyncMetrics: Synchronization metrics
        """
        metrics = SyncMetrics()

        if is_actor and self._weights_info is None:
            raise ValueError("weights_info must be set before sync for actor")

        # 如果使用检查点引擎，使用优化后的同步路径
        if use_checkpoint_engine and self.inference_backend_type != "sglang":
            return self._sync_weights_by_checkpoint(
                model, inference_model, is_actor, is_rollout, params
            )

        # 标准同步路径
        if self.inference_backend_type == "vllm":
            return self._sync_vllm_weights(
                model, inference_model, is_actor, is_rollout, params
            )
        elif self.inference_backend_type == "sglang":
            return self._sync_sglang_weights(
                model, inference_model, is_actor, is_rollout, params
            )

    def _sync_vllm_weights(
        self,
        model,
        inference_model,
        is_actor: bool,
        is_rollout: bool,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> SyncMetrics:
        """同步权重到 vLLM"""
        metrics = SyncMetrics()
        start_time = time.time()

        # 加载模型到 GPU（如果需要）
        if is_actor:
            self.model_backend.load_model_to_gpu(model)

        # 收集参数（优先使用传入的 params，其次使用 engine 获取）
        if is_actor and params is None:
            if self._engine is not None:
                params = self._engine.get_per_tensor_param()
            else:
                raise ValueError("params required for actor sync (pass engine or params)")

        # 同步每个权重
        for weight_info in self._weights_info:
            tensor = torch.empty(
                weight_info.shape,
                dtype=weight_info.dtype,
                device=get_torch_device().current_device(),
            )

            if is_actor:
                assert weight_info.key in params
                origin_data = params[weight_info.key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            # 广播
            broadcast_tensor(tensor, src_rank=0, group_name=self._sync_group_name, sync_group=self._sync_group)

            # 加载到推理模型
            if is_rollout:
                self.inference_backend.load_weights(inference_model, [(weight_info.key, tensor)])

        metrics.sync_time = time.time() - start_time

        # 卸载模型回 CPU（如果需要）
        if is_actor:
            self.model_backend.offload_model_to_cpu(model)

        return metrics

    def _sync_sglang_weights(
        self,
        model,
        inference_model,
        is_actor: bool,
        is_rollout: bool,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> SyncMetrics:
        """同步权重到 SGLang（使用分桶优化）"""
        metrics = SyncMetrics()
        start_time = time.time()

        # 加载模型到 GPU（如果需要）
        if is_actor:
            self.model_backend.load_model_to_gpu(model)

        # 收集参数（优先使用传入的 params，其次使用 engine 获取）
        if is_actor and params is None:
            if self._engine is not None:
                params = self._engine.get_per_tensor_param()
            else:
                raise ValueError("params required for actor sync (pass engine or params)")

        # 分桶同步
        bucket_size_bytes = int(self._bucket_size_mb * 1024 * 1024)
        actual_bucket_sizes: List[float] = []
        current_batch: List[Tuple[str, torch.Tensor]] = []
        current_batch_size = 0

        def flush_batch():
            if current_batch:
                actual_bucket_sizes.append(current_batch_size / (1024 * 1024))
                self._run_async_safely(
                    self.inference_backend.update_weights(
                        inference_model,
                        iter(current_batch),
                        "infer_tp",
                        self._device_mesh,
                    )
                )
                get_torch_device().synchronize()
                current_batch.clear()

        for weight_info in self._weights_info:
            tensor = torch.empty(
                weight_info.shape,
                dtype=weight_info.dtype,
                device=get_torch_device().current_device(),
            )

            if is_actor:
                assert weight_info.key in params
                origin_data = params[weight_info.key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            # 广播
            broadcast_tensor(tensor, src_rank=0, group_name=self._sync_group_name, sync_group=self._sync_group)

            # 添加到当前批次
            current_batch.append((weight_info.key, tensor))
            current_batch_size += tensor.numel() * tensor.element_size()

            # 检查是否需要 flush
            if current_batch_size >= bucket_size_bytes:
                flush_batch()
                current_batch_size = 0

        # Flush 剩余批次
        flush_batch()

        # 更新指标
        metrics.sync_time = time.time() - start_time
        if actual_bucket_sizes:
            metrics.bucket_size_mb = sum(actual_bucket_sizes) / len(actual_bucket_sizes)

        # 更新分桶大小
        self._update_bucket_size(metrics.bucket_size_mb, metrics.sync_time)

        # 恢复 KV cache（仅在 rollout rank 0）
        if is_rollout and self._device_mesh["infer_tp"].get_local_rank() == 0:
            self._run_async_safely(self.inference_backend.resume_memory_occupation(inference_model))

        # 卸载模型回 CPU（如果需要）
        if is_actor:
            self.model_backend.offload_model_to_cpu(model)

        return metrics

    def _sync_weights_by_checkpoint(
        self,
        model,
        inference_model,
        is_actor: bool,
        is_rollout: bool,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> SyncMetrics:
        """使用检查点引擎同步权重（优化版本）"""
        metrics = SyncMetrics()
        start_time = time.time()

        # 加载模型到 GPU
        load_start = time.time()
        if is_actor:
            self.model_backend.load_model_to_gpu(model)
        metrics.cache_time = time.time() - load_start

        # 收集并缓存权重到 CPU
        cache_start = time.time()
        if is_actor:
            cpu_params = self._cache_params_to_cpu(model, params)
        metrics.cache_time += time.time() - cache_start

        # 注册到检查点引擎
        register_start = time.time()
        if self._checkpoint_engine is not None:
            self._checkpoint_engine.register_checkpoint(self._weights_info, cpu_params)
        metrics.register_time = time.time() - register_start

        # 等待所有 rank 准备就绪
        collective.barrier(group_name=self._sync_group_name)

        # 更新检查点
        update_start = time.time()
        if self._checkpoint_engine is not None and is_rollout:
            self._checkpoint_engine.update_checkpoint(
                inference_model=inference_model,
                group_name=self._sync_group_name,
                overlap_broadcast_and_consume=True,
            )
        metrics.update_time = time.time() - update_start

        # 卸载模型回 CPU
        offload_start = time.time()
        if is_actor:
            self.model_backend.offload_model_to_cpu(model)
        metrics.cache_time += time.time() - offload_start

        metrics.sync_time = time.time() - start_time

        return metrics

    def _cache_params_to_cpu(
        self,
        model,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """缓存参数到 CPU"""
        cpu_params = {}

        # 优先使用传入的 params，其次使用 engine 获取
        if params is None:
            if self._engine is not None:
                params = self._engine.get_per_tensor_param()
            else:
                raise ValueError("params required for caching (pass engine or params)")

        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        for idx, weight_info in enumerate(self._weights_info):
            if idx % world_size == local_rank:
                origin_data = params[weight_info.key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                cpu_params[weight_info.key] = origin_data.to("cpu", non_blocking=True)

        get_torch_device().synchronize()
        return cpu_params

    def _update_bucket_size(self, bucket_size_mb: float, sync_time: float) -> None:
        """动态调整分桶大小"""
        self._sync_history.append((bucket_size_mb, sync_time))
        if len(self._sync_history) > self._max_history_size:
            self._sync_history.pop(0)

        if len(self._sync_history) < 4:
            self._bucket_size_mb = min(self._max_bucket_size_mb, self._bucket_size_mb * 1.5)
        else:
            times = [t for _, t in self._sync_history[-4:]]
            buckets = [b for b, _ in self._sync_history[-4:]]

            recent_avg_time = sum(times[-2:]) / 2
            previous_avg_time = sum(times[:2]) / 2
            recent_avg_bucket = sum(buckets[-2:]) / 2
            previous_avg_bucket = sum(buckets[:2]) / 2

            performance_improved = recent_avg_time < previous_avg_time
            bucket_increased = recent_avg_bucket > previous_avg_bucket

            if recent_avg_time < previous_avg_time * 0.8:
                step = 1.2
            elif recent_avg_time < previous_avg_time * 0.9:
                step = 1.1
            elif recent_avg_time < previous_avg_time * 0.95:
                step = 1.05
            else:
                step = 1.02

            should_increase = (performance_improved and bucket_increased) or (
                not performance_improved and not bucket_increased
            )

            if should_increase:
                self._bucket_size_mb = min(
                    self._max_bucket_size_mb,
                    self._bucket_size_mb * step,
                )
            else:
                self._bucket_size_mb = max(
                    self._min_bucket_size_mb,
                    self._bucket_size_mb / step,
                )

    # ==================== 外部接口 ====================

    def set_weights_info(self, weights_info: List[WeightInfo]) -> None:
        """设置权重信息"""
        self._weights_info = weights_info

    def get_weights_info(self) -> Optional[List[WeightInfo]]:
        """获取权重信息"""
        return self._weights_info

    def set_device_mesh(self, device_mesh: Dict) -> None:
        """设置设备网格（用于 SGLang）"""
        self._device_mesh = device_mesh

    def set_checkpoint_engine(self, checkpoint_engine) -> None:
        """设置检查点引擎"""
        self._checkpoint_engine = checkpoint_engine

    def cleanup(self) -> None:
        """清理资源"""
        if hasattr(self, "_bg_loop") and self._bg_loop.is_running():
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if hasattr(self, "_bg_thread") and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)


# ==================== 工厂函数 ====================

def create_weight_synchronizer(
    model_backend: str,
    inference_backend: str,
    device: Optional[str] = None,
) -> UnifiedWeightSynchronizer:
    """
    创建权重同步器的工厂函数

    Args:
        model_backend: "fsdp" or "megatron"
        inference_backend: "vllm" or "sglang"
        device: "gpu" or "npu" (auto-detected if None)

    Returns:
        UnifiedWeightSynchronizer instance
    """
    if device is None:
        device = "npu" if is_npu_available else "gpu"

    return UnifiedWeightSynchronizer(
        model_backend=model_backend,
        inference_backend=inference_backend,
        device=device,
    )


# ==================== 工具函数 ====================

def patch_vllm_moe_weight_loader(inference_model) -> None:
    """Patch vLLM MoE model weight loader"""
    from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

    patch_vllm_moe_model_weight_loader(inference_model)


def create_sync_group_for_ranks(
    workers: List,
    world_size: int,
    group_name: str = "actor_rollout",
) -> None:
    """
    为指定 workers 创建同步组

    Args:
        workers: List of workers
        world_size: World size
        group_name: Group name
    """
    from verl.utils.device import get_nccl_backend

    collective.create_collective_group(
        workers,
        world_size,
        list(range(world_size)),
        backend=get_nccl_backend(),
        group_name=group_name,
    )


# ==================== 统一参数同步管理器 ====================

class UnifiedParamSyncManager:
    """
    统一参数同步管理器基类

    封装所有参数同步逻辑，Worker 只需继承并提供必要信息。
    支持:
    - 训练框架: FSDP (v1/v2), Megatron
    - 推理引擎: vLLM, SGLang
    - 硬件设备: GPU, NPU
    - 同步方式: 标准同步、检查点引擎同步

    使用方式:
        1. Worker 继承此类
        2. 实现 _get_model_info() 和 _get_inference_model() 方法
        3. 调用 sync_weights() 进行同步
    """

    # 动态分桶配置
    _bucket_size_mb: float = 1024.0
    _max_bucket_size_mb: float = 8192.0
    _min_bucket_size_mb: float = 512.0
    _sync_history: List[Tuple[float, float]] = []
    _max_history_size: int = 20

    def __init__(self, config=None, role: str = None):
        """
        初始化同步管理器

        Args:
            config: 配置对象
            role: 角色名称
        """
        self.config = config
        self.role = role

        # 同步状态
        self._weights_info: Optional[List] = None
        self._sync_group_name: str = "actor_rollout"
        self._weight_sync_group = None

        # 背景事件循环（用于 SGLang）
        self._bg_loop = None
        self._bg_thread = None

        # 检查点引擎
        self._checkpoint_engine = None

    # ==================== 子类需实现的方法 ====================

    @property
    @abstractmethod
    def is_actor(self) -> bool:
        """是否为 Actor"""
        pass

    @property
    @abstractmethod
    def is_rollout(self) -> bool:
        """是否为 Rollout"""
        pass

    @property
    @abstractmethod
    def is_offload_param(self) -> bool:
        """是否需要 offload 参数"""
        pass

    @property
    @abstractmethod
    def rollout_name(self) -> str:
        """Rollout 名称 (vllm/sglang)"""
        pass

    def _get_model(self):
        """获取训练模型，子类可重写"""
        if self.is_actor:
            return getattr(self, 'actor_module_fsdp', None) or getattr(self, 'actor_module', None)
        return None

    def _get_rollout(self):
        """获取 rollout 对象，子类可重写"""
        return getattr(self, 'rollout', None)

    def _get_actor_params_generator(self):
        """
        获取 actor 参数生成器（Megatron 使用）
        子类可重写
        """
        return None

    def _init_sglang_engine(self):
        """
        初始化 SGLang 引擎（用于 ServerAdapter）
        子类可重写
        """
        rollout = self._get_rollout()
        if rollout is None:
            return None

        async def init_engine():
            if hasattr(rollout, "_init_server_adapter"):
                await rollout._init_server_adapter()
            else:
                print("[_init_sglang_engine] No _init_server_adapter method found")
            return rollout._engine

        return self._run_async_safely(init_engine())

    # ==================== 权重信息管理 ====================

    def set_weights_info(self, weights_info: List) -> None:
        """设置权重信息"""
        self._weights_info = weights_info

    def get_weights_info(self) -> Optional[List]:
        """获取权重信息"""
        return self._weights_info

    def _build_weights_info(self, params: Dict[str, torch.Tensor]) -> List:
        """从参数字典构建权重信息"""
        return [(key, tensor.size(), tensor.dtype) for key, tensor in params.items()]

    # ==================== 同步组管理 ====================

    def create_weight_sync_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        """
        创建权重同步组

        Args:
            master_address: Master 节点地址
            master_port: Master 节点端口
            rank_offset: Rank 偏移
            world_size: 世界大小
        """
        rank = torch.distributed.get_rank() + rank_offset

        if is_npu_available:
            self._weight_sync_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                get_torch_device().current_device(),
            )

    # ==================== 核心同步方法 ====================

    def sync_weights(self, sync_group_name: str = "actor_rollout") -> Dict:
        """
        执行权重同步（统一入口）

        Args:
            sync_group_name: 同步组名称

        Returns:
            Dict: 同步指标
        """
        assert (self.is_actor or self.is_rollout) and not self.config.hybrid_engine
        assert self._weights_info is not None

        self._sync_group_name = sync_group_name

        # 决定使用哪种同步方式
        use_checkpoint = (
            hasattr(self, 'checkpoint_engine')
            and self.checkpoint_engine is not None
            and self.rollout_name != "sglang"
        )

        if use_checkpoint:
            return self._sync_by_checkpoint()
        elif self.rollout_name == "sglang":
            return self._sync_sglang()
        else:
            return self._sync_vllm()

    def _sync_vllm(self) -> Dict:
        """同步权重到 vLLM"""
        metrics = {"sync_time": 0.0}
        start_time = time.time()

        # 1. 加载模型到 GPU
        if self.is_actor and self.is_offload_param:
            self._load_model_to_gpu()

        # 2. 获取参数
        params = self._get_actor_params() if self.is_actor else None

        # 3. 获取推理模型
        inference_model = None
        if self.is_rollout:
            inference_model = self._get_inference_model()
            if inference_model is not None:
                patch_vllm_moe_weight_loader(inference_model)

        # 4. 执行同步
        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())

            if self.is_actor:
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            # 广播
            self._broadcast_tensor(tensor, src_rank=0)

            # 加载到推理模型
            if self.is_rollout and inference_model is not None:
                inference_model.load_weights([(key, tensor)])

        # 5. 卸载模型回 CPU
        if self.is_actor and self.is_offload_param:
            self._offload_model_to_cpu()

        metrics["sync_time"] = time.time() - start_time
        get_torch_device().empty_cache()

        return metrics

    def _sync_sglang(self) -> Dict:
        """同步权重到 SGLang（使用分桶优化）"""
        metrics = {"sync_time": 0.0, "bucket_sizes": []}
        start_time = time.time()

        # 1. 加载模型到 GPU
        if self.is_actor and self.is_offload_param:
            self._load_model_to_gpu()

        # 2. 获取参数
        params = self._get_actor_params() if self.is_actor else None
        params_generator = self._get_actor_params_generator()

        # 3. 获取推理模型
        inference_model = None
        if self.is_rollout:
            inference_model = self._get_rollout()._engine
            if inference_model is None:
                inference_model = self._init_sglang_engine()

        # 4. 分桶同步
        bucket_size_bytes = int(self._bucket_size_mb * 1024 * 1024)
        actual_bucket_sizes: List[float] = []
        current_batch = []
        current_batch_size = 0

        def flush_batch():
            if current_batch:
                actual_bucket_sizes.append(current_batch_size / (1024 * 1024))
                self._run_async_safely(
                    self._sglang_update_weights(inference_model, iter(current_batch))
                )
                get_torch_device().synchronize()
                current_batch.clear()

        for key, shape, dtype in self._weights_info:
            # Megatron 使用生成器
            if self.is_actor:
                if params_generator is not None:
                    weight_key, weight = next(params_generator)
                    assert key == weight_key
                else:
                    weight = params[key]

            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())

            if self.is_actor and torch.distributed.get_rank() == 0:
                if params_generator is not None:
                    tensor.copy_(weight)
                else:
                    origin_data = params[key]
                    if hasattr(origin_data, "full_tensor"):
                        origin_data = origin_data.full_tensor()
                    tensor.copy_(origin_data)

            # 广播
            self._broadcast_tensor(tensor, src_rank=0)

            # 添加到批次
            current_batch.append((key, tensor))
            current_batch_size += tensor.numel() * tensor.element_size()

            # 检查是否需要 flush
            if current_batch_size >= bucket_size_bytes:
                flush_batch()
                current_batch_size = 0

        # Flush 剩余批次
        flush_batch()

        # 5. 更新分桶大小
        if actual_bucket_sizes:
            avg_bucket = sum(actual_bucket_sizes) / len(actual_bucket_sizes)
            self._update_bucket_size(avg_bucket, 0)
            metrics["bucket_sizes"] = actual_bucket_sizes

        # 6. 恢复 KV cache
        if self.is_rollout:
            rollout_device_mesh = getattr(self, 'rollout_device_mesh', None)
            if rollout_device_mesh and rollout_device_mesh["infer_tp"].get_local_rank() == 0:
                self._run_async_safely(inference_model.resume_memory_occupation(tags=["kv_cache"]))

        # 7. 卸载模型回 CPU
        if self.is_actor and self.is_offload_param:
            self._offload_model_to_cpu()

        metrics["sync_time"] = time.time() - start_time
        get_torch_device().empty_cache()

        return metrics

    def _sync_by_checkpoint(self) -> Dict:
        """使用检查点引擎同步权重"""
        from ray.util.collective import collective

        metrics = {"cache_time": 0.0, "register_time": 0.0, "update_time": 0.0}
        start_time = time.time()

        # 1. 加载模型到 GPU
        if self.is_actor and self.is_offload_param:
            self._load_model_to_gpu()

        # 2. 缓存权重到 CPU
        cache_start = time.time()
        cpu_params = self._cache_params_to_cpu()
        metrics["cache_time"] = time.time() - cache_start

        # 3. 注册到检查点引擎
        register_start = time.time()
        self._checkpoint_engine.register_checkpoint(self._weights_info, cpu_params)
        metrics["register_time"] = time.time() - register_start

        # 4. 等待所有 rank
        collective.barrier(group_name=self._sync_group_name)

        # 5. 更新检查点
        update_start = time.time()
        inference_model = None
        if self.is_rollout:
            inference_model = self._get_inference_model()
            if inference_model is not None:
                patch_vllm_moe_weight_loader(inference_model)

        self._checkpoint_engine.update_checkpoint(
            inference_model=inference_model,
            group_name=self._sync_group_name,
            overlap_broadcast_and_consume=getattr(self.config.checkpoint_engine, 'overlap_broadcast_and_consume', True),
        )
        metrics["update_time"] = time.time() - update_start

        # 6. 卸载模型回 CPU
        if self.is_actor and self.is_offload_param:
            self._offload_model_to_cpu()

        get_torch_device().empty_cache()

        return metrics

    # ==================== 统一参数获取接口 ====================

    def get_per_tensor_param(self) -> Dict[str, torch.Tensor]:
        """
        统一获取 Actor 参数的方法

        自动检测模型类型并调用对应的后端:
        - FSDP: 使用 FSDPBackend.get_params()
        - Megatron: 使用 MegatronBackend.get_params_generator() 并转换为字典

        Returns:
            Dict[str, torch.Tensor]: 参数字典
        """
        model = self._get_model()
        if model is None:
            return {}

        # 尝试检测模型类型并调用对应的后端
        # 1. 优先检查是否有 _get_actor_params_custom 方法（兼容旧接口）
        if hasattr(self, '_get_actor_params_custom'):
            return self._get_actor_params_custom()

        # 2. 检测 FSDP 模型
        if self._is_fsdp_model(model):
            return FSDPBackend.get_params(model)

        # 3. 检测 Megatron 模型
        if self._is_megatron_model(model):
            # Megatron 使用生成器，需要转换为字典
            params_generator = self._get_actor_params_generator()
            if params_generator is not None:
                return dict(params_generator)
            # 如果没有生成器，尝试直接获取
            return {}

        # 4. 默认实现：使用 state_dict
        if hasattr(model, 'state_dict'):
            from verl.utils.model import convert_weight_keys
            params = model.state_dict()
            params = convert_weight_keys(params, getattr(model, "_fsdp_wrapped_module", model))
            return params

        return {}

    def _is_fsdp_model(self, model) -> bool:
        """检测是否为 FSDP 模型"""
        try:
            from verl.utils.fsdp_utils import fsdp_version
            fsdp_version(model)
            return True
        except (ImportError, Exception):
            return False

    def _is_megatron_model(self, model) -> bool:
        """检测是否为 Megatron 模型"""
        # Megatron 模型通常有特定的结构特征
        try:
            import megatron.core
            return True
        except ImportError:
            return False

    # ==================== 内部辅助方法 ====================

    def _get_actor_params(self) -> Dict[str, torch.Tensor]:
        """获取 Actor 参数"""
        return self.get_per_tensor_param()

    # ==================== 统一模型加载/卸载接口 ====================

    def load_model_to_gpu(self) -> None:
        """
        统一加载模型到 GPU 的方法

        自动检测模型类型并调用对应的后端:
        - FSDP: 使用 FSDPBackend.load_model_to_gpu()
        - Megatron: 使用 MegatronBackend.load_model_to_gpu()
        """
        model = self._get_model()
        if model is None:
            return

        # 优先检查是否有 _load_model_to_gpu_custom 方法（兼容旧接口）
        if hasattr(self, '_load_model_to_gpu_custom'):
            self._load_model_to_gpu_custom()
            return

        # 尝试加载模型
        self._try_load_model_to_gpu(model)

    def offload_model_to_cpu(self) -> None:
        """
        统一卸载模型到 CPU 的方法

        自动检测模型类型并调用对应的后端:
        - FSDP: 使用 FSDPBackend.offload_model_to_cpu()
        - Megatron: 使用 MegatronBackend.offload_model_to_cpu()
        """
        model = self._get_model()
        if model is None:
            return

        # 优先检查是否有 _offload_model_to_cpu_custom 方法（兼容旧接口）
        if hasattr(self, '_offload_model_to_cpu_custom'):
            self._offload_model_to_cpu_custom()
            return

        # 尝试卸载模型
        self._try_offload_model_to_cpu(model)

    def _try_load_model_to_gpu(self, model) -> bool:
        """尝试加载模型到 GPU，返回是否成功"""
        # 1. 尝试 FSDP
        if self._is_fsdp_model(model):
            FSDPBackend.load_model_to_gpu(model)
            return True

        # 2. 尝试 Megatron
        if self._is_megatron_model(model):
            MegatronBackend.load_model_to_gpu(model)
            return True

        return False

    def _try_offload_model_to_cpu(self, model) -> bool:
        """尝试卸载模型到 CPU，返回是否成功"""
        # 1. 尝试 FSDP
        if self._is_fsdp_model(model):
            FSDPBackend.offload_model_to_cpu(model)
            return True

        # 2. 尝试 Megatron
        if self._is_megatron_model(model):
            MegatronBackend.offload_model_to_cpu(model)
            return True

        return False

    # ==================== 内部辅助方法 ====================

    def _load_model_to_gpu(self) -> None:
        """加载模型到 GPU（兼容旧接口）"""
        self.load_model_to_gpu()

    def _offload_model_to_cpu(self) -> None:
        """卸载模型到 CPU（兼容旧接口）"""
        self.offload_model_to_cpu()

    def _get_inference_model(self) -> Optional[torch.nn.Module]:
        """获取推理模型"""
        rollout = self._get_rollout()
        if rollout is None:
            return None
        return get_inference_model(rollout)

    def _broadcast_tensor(self, tensor: torch.Tensor, src_rank: int = 0) -> None:
        """广播张量"""
        if is_npu_available:
            if self._weight_sync_group is not None:
                self._weight_sync_group.broadcast(tensor, src=src_rank, stream=get_torch_device().current_stream())
        else:
            collective.broadcast(tensor, src_rank=src_rank, group_name=self._sync_group_name)

    def _cache_params_to_cpu(self) -> Dict[str, torch.Tensor]:
        """缓存参数到 CPU"""
        params = self._get_actor_params()
        cpu_params = {}

        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        for idx, (key, shape, dtype) in enumerate(self._weights_info):
            if idx % world_size == local_rank:
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                cpu_params[key] = origin_data.to("cpu", non_blocking=True)

        get_torch_device().synchronize()
        return cpu_params

    async def _sglang_update_weights(self, inference_engine, params_batch) -> None:
        """SGLang 异步更新权重"""
        await SGLangBackend.update_weights(
            inference_engine=inference_engine,
            params_batch=params_batch,
            device_mesh_key="infer_tp",
            device_mesh=getattr(self, 'rollout_device_mesh', None),
        )
        if getattr(self, 'rollout_device_mesh', {}).get("infer_tp", {}).get_local_rank() == 0:
            await SGLangBackend.flush_cache(inference_engine)

    def _update_bucket_size(self, bucket_size_mb: float, sync_time: float) -> None:
        """动态调整分桶大小"""
        self._sync_history.append((bucket_size_mb, sync_time))
        if len(self._sync_history) > self._max_history_size:
            self._sync_history.pop(0)

        if len(self._sync_history) < 4:
            self._bucket_size_mb = min(self._max_bucket_size_mb, self._bucket_size_mb * 1.5)
        else:
            times = [t for _, t in self._sync_history[-4:]]
            buckets = [b for b, _ in self._sync_history[-4:]]

            recent_avg_time = sum(times[-2:]) / 2
            previous_avg_time = sum(times[:2]) / 2
            recent_avg_bucket = sum(buckets[-2:]) / 2
            previous_avg_bucket = sum(buckets[:2]) / 2

            performance_improved = recent_avg_time < previous_avg_time
            bucket_increased = recent_avg_bucket > previous_avg_bucket

            step = 1.1 if recent_avg_time < previous_avg_time * 0.9 else 1.02
            should_increase = (performance_improved and bucket_increased) or (
                not performance_improved and not bucket_increased
            )

            if should_increase:
                self._bucket_size_mb = min(self._max_bucket_size_mb, self._bucket_size_mb * step)
            else:
                self._bucket_size_mb = max(self._min_bucket_size_mb, self._bucket_size_mb / step)

    # ==================== 背景事件循环 ====================

    def _init_background_loop(self) -> None:
        """初始化背景事件循环"""
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._start_background_loop,
            args=(self._bg_loop,),
            name="param_sync_async_worker",
            daemon=True,
        )
        self._bg_thread.start()

    def _start_background_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """启动背景事件循环"""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"[UnifiedParamSyncManager] Background loop crashed: {e}")

    def _run_async_safely(self, coro) -> Any:
        """安全运行异步协程"""
        if self._bg_loop is None or not self._bg_thread.is_alive():
            self._init_background_loop()

        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return future.result()

    # ==================== 检查点引擎接口 ====================

    def init_checkpoint_engine(self, rank_offset: int, actor_num: int, rollout_num: int) -> None:
        """初始化检查点引擎"""
        from .checkpoint_engine import CheckpointEngine

        current_rank = torch.distributed.get_rank() + rank_offset
        actor_ranks = list(range(actor_num))
        rollout_ranks = [rank + actor_num for rank in range(rollout_num)]

        self._checkpoint_engine = CheckpointEngine(
            current_rank,
            actor_ranks,
            rollout_ranks,
            getattr(self.config.checkpoint_engine, 'device_buffer_size_M', 1024),
        )

    # ==================== 指标记录 ====================

    @classmethod
    def get_bucket_size_mb(cls) -> float:
        return cls._bucket_size_mb

    @classmethod
    def record_sync_metrics(cls, bucket_size_mb: float, sync_time: float) -> None:
        """记录同步指标"""
        cls._sync_history.append((bucket_size_mb, sync_time))
        if len(cls._sync_history) > cls._max_history_size:
            cls._sync_history.pop(0)

    def save_model_to_cpu(self, key: str) -> None:
        """保存模型到 CPU（用于 FSDP2）"""
        model = self._get_model()
        if model is None:
            return

        if not hasattr(self, 'cpu_saved_models'):
            self.cpu_saved_models = {}

        try:
            from verl.experimental.fully_async_policy.fsdp2_utils import fsdp2_sharded_save_to_cpu
            self.cpu_saved_models[key] = fsdp2_sharded_save_to_cpu(model)
        except ImportError:
            pass

    def restore_model_from_cpu(self, key: str) -> None:
        """从 CPU 恢复模型"""
        model = self._get_model()
        if model is None or not hasattr(self, 'cpu_saved_models'):
            return

        if key in self.cpu_saved_models:
            try:
                from verl.experimental.fully_async_policy.fsdp2_utils import fsdp2_sharded_load_from_cpu
                cpu_sharded_state, global_spec = self.cpu_saved_models[key]
                fsdp2_sharded_load_from_cpu(model, cpu_sharded_state, global_spec)
            except ImportError:
                pass

    def clear_cpu_model(self, key: str) -> None:
        """清除 CPU 上的模型"""
        if hasattr(self, 'cpu_saved_models') and key in self.cpu_saved_models:
            del self.cpu_saved_models[key]

