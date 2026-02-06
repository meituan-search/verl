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

from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from verl.workers.engine import MegatronEngine


class SeparateMegatronEngine(MegatronEngine):
    # ==================== 模型加载/卸载（用于参数同步） ====================

    def load_model_to_gpu(self, load_grad: bool = False):
        """
        将 Megatron 模型加载到 GPU（用于参数同步）

        Args:
            load_grad: 是否同时加载梯度
        """
        load_megatron_model_to_gpu(self.module, load_grad=load_grad)

    def offload_model_to_cpu(self):
        """
        将 Megatron 模型卸载到 CPU（用于参数同步）
        """
        offload_megatron_model_to_cpu(self.module)
