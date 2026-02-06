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

from torch.distributed.tensor import DTensor

from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.engine import FSDPEngine


class SeparateFSDPEngine(FSDPEngine):
    # ==================== 模型加载/卸载（用于参数同步） ====================

    def load_model_to_gpu(self):
        """
        将 FSDP 模型加载到 GPU（用于参数同步）
        """
        load_fsdp_model_to_gpu(self.module)

    def offload_model_to_cpu(self):
        """
        将 FSDP 模型卸载到 CPU（用于参数同步）
        """
        offload_fsdp_model_to_cpu(self.module)

    # ==================== FSDP2 模型 CPU 保存/恢复 ====================

    def save_model_to_cpu(self, n: str) -> tuple[dict, DTensor]:
        """
        保存 FSDP2 模型到 CPU 内存（用于异步训练时的模型交换）

        Args:
            n: 标识符，用于区分不同的保存副本

        Returns:
            cpu_sharded_state: CPU 上的分片状态字典
            global_spec: 全局 DTensorSpec
        """
        from verl.experimental.fully_async_policy.fsdp2_utils import fsdp2_sharded_save_to_cpu

        if not hasattr(self, "_cpu_saved_models"):
            self._cpu_saved_models = {}

        cpu_sharded_state, global_spec = fsdp2_sharded_save_to_cpu(self.module)
        self._cpu_saved_models[n] = (cpu_sharded_state, global_spec)
        return cpu_sharded_state, global_spec

    def restore_model_from_cpu(self, n: str) -> bool:
        """
        从 CPU 内存恢复 FSDP2 模型

        Args:
            n: 标识符，指定要恢复的保存副本

        Returns:
            bool: 是否成功恢复
        """
        if n not in self._cpu_saved_models:
            return False

        from verl.experimental.fully_async_policy.fsdp2_utils import fsdp2_sharded_load_from_cpu

        cpu_sharded_state, global_spec = self._cpu_saved_models[n]
        fsdp2_sharded_load_from_cpu(self.module, cpu_sharded_state, global_spec)
        return True

    def clear_cpu_model(self, n: str) -> bool:
        """
        清除 CPU 上保存的模型副本

        Args:
            n: 标识符，指定要清除的保存副本

        Returns:
            bool: 是否成功清除
        """
        if n not in self._cpu_saved_models:
            return False

        del self._cpu_saved_models[n]
        return True

    def has_cpu_model(self, n: str) -> bool:
        """
        检查是否在 CPU 上有保存的模型副本

        Args:
            n: 标识符

        Returns:
            bool: 是否存在
        """
        return hasattr(self, "_cpu_saved_models") and n in self._cpu_saved_models
