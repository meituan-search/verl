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

# ==================== 统一参数同步 Worker 基类 ====================
# 用于 FSDP/Megatron + vLLM/SGLang 的参数同步，支持 GPU 和 NPU
from abc import ABC

from omegaconf import DictConfig

from verl.experimental.separation.param_sync import UnifiedParamSyncManager
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


class DetachAsyncRolloutWorker(ActorRolloutRefWorker, UnifiedParamSyncManager, ABC):
    """
    统一参数同步 Worker 基类

    统一支持 FSDP/Megatron + vLLM/SGLang 的参数同步
    支持 GPU 和 NPU

    使用方式:
        1. 对于 FSDP: 继承 DetachAsyncRolloutWorker 和 FSDP Worker 基类
        2. 对于 Megatron: 继承 DetachAsyncRolloutWorker 和 Megatron Worker 基类
        3. 实现 _get_model() 和 _get_rollout() 方法提供模型和 rollout 引用
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        # 先初始化 ActorRolloutRefWorker，提供 actor_module、actor 等属性
        ActorRolloutRefWorker.__init__(self, config, role, **kwargs)
        # 再初始化 UnifiedParamSyncManager，提供同步功能
        UnifiedParamSyncManager.__init__(self, config, role)

    @property
    def is_offload_param(self) -> bool:
        """是否需要 offload 参数"""
        return self._is_offload_param

    @property
    def rollout_name(self) -> str:
        """Rollout 名称 (vllm/sglang)"""
        return self.config.rollout.name

    def _get_model(self):
        """获取训练模型，子类需重写"""
        return None

    def _get_rollout(self):
        """获取 rollout 对象，子类需重写"""
        return None

    def _get_actor_params_generator(self):
        """
        获取 actor 参数生成器（Megatron 使用）
        子类可重写
        """
        return None

    # ==================== 同步方法 ====================

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self, sync_group_name: str = "actor_rollout"):
        """同步权重到 rollout（使用统一同步逻辑）"""
        self.sync_weights(sync_group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_checkpoint(self, sync_group_name: str = "actor_rollout"):
        """使用检查点引擎同步权重"""
        self.sync_weights(sync_group_name)


class DetachActorWorker(DetachAsyncRolloutWorker):
    """Actor Worker，使用统一的参数同步逻辑"""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        """获取 Actor 权重信息"""
        assert self.is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info

        # 加载模型到 GPU（如果需要）
        if self.is_offload_param:
            self._load_model_to_gpu()

        # 获取参数
        params = self._get_actor_params()
        if params:
            self._weights_info = [(key, tensor.size(), tensor.dtype) for key, tensor in params.items()]
        else:
            self._weights_info = []

        return self._weights_info


class DetachRolloutWorker(DetachAsyncRolloutWorker):
    """Rollout Worker，用于接收同步的权重"""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        """设置从 actor 同步过来的权重信息"""
        assert self.is_rollout
        self._weights_info = weights_info
