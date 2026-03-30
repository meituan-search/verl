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
Elastic Megatron Engine for dynamic DP group resizing.

Provides:
- ElasticMegatronEngine: Base elastic engine with DP rebuild capability
- ElasticMegatronEngineWithLMHead: LM head variant with dynamic DP support

Key features:
1. DP group rebuild without disk I/O (state stored in CPU memory)
2. Support for adding/removing training resources
3. Parameter synchronization for newly joined ranks
4. Seamless integration with existing MegatronEngine interface
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from verl.utils.device import get_device_name
from verl.workers.engine import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import (
    MegatronEngineWithLMHead,
    MegatronEngineWithValueHead,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ElasticMegatronMixin:
    """
    Mixin that adds elastic DP group rebuild capabilities to any Megatron engine.

    This mixin provides the rebuild_dp_group method that handles:
    1. Capturing model and optimizer state to CPU
    2. Destroying and reinitializing parallel groups
    3. Restoring state without disk I/O
    4. Syncing parameters to newly joined ranks

    Usage:
        class MyElasticEngine(MegatronEngineWithLMHead, ElasticMegatronMixin):
            pass
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State snapshots for DP rebuild
        self._model_snapshot: Optional[ModelStateSnapshot] = None
        self._optimizer_snapshot: Optional[OptimizerStateSnapshot] = None

        # Track device placement
        self._model_on_gpu = True
        self._optimizer_on_gpu = True

    def rebuild_dp_group(
        self,
        new_world_ranks: list[int],
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
    ) -> None:
        """
        Rebuild DP group for elastic training.

        Procedure:
        1. Capture model and optimizer state to CPU
        2. Offload everything to CPU
        3. Destroy all parallel groups
        4. Barrier with all ranks (including new ones)
        5. Reinitialize parallel groups
        6. Restore state from CPU
        7. Sync parameters to newly joined ranks

        Args:
            new_world_ranks: List of global ranks in new DP group
            tensor_model_parallel_size: TP size (usually unchanged)
            pipeline_model_parallel_size: PP size (usually unchanged)
            context_parallel_size: CP size
            expert_model_parallel_size: EP size
            expert_tensor_parallel_size: Expert TP size
        """
        my_rank = dist.get_rank()

        # Check if this rank is in new DP group
        is_in_new_group = my_rank in new_world_ranks

        logger.info(
            f"[ElasticMegatronMixin rank={my_rank}] rebuild_dp_group called\n"
            f"  new_world_ranks={new_world_ranks}\n"
            f"  tp={tensor_model_parallel_size}, pp={pipeline_model_parallel_size}\n"
            f"  is_in_new_group={is_in_new_group}"
        )

        try:
            # Step 1: Capture state to CPU
            self._capture_state_to_cpu()

            # Step 2: Destroy old parallel groups
            self._destroy_parallel_groups()

            # Step 3: Collective — ALL ranks must participate in new_group and
            # initialize_model_parallel (both call dist.new_group internally).
            # We do this BEFORE the early-return so that ranks removed from the
            # new group still execute the same number of collectives.
            #
            # 3a. Pre-create the sync group (used later for param broadcast).
            #     Every rank participates regardless of is_in_new_group.
            sync_group = dist.new_group(ranks=new_world_ranks)

            # 3b. Reinitialize Megatron parallel groups (contains dist.new_group
            #     calls; ranks not in new_world_ranks still need to participate).
            self._reinitialize_parallel_groups(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
                expert_tensor_parallel_size=expert_tensor_parallel_size,
            )

            # Step 4: Global barrier — all ranks synchronize after collectives.
            if dist.is_initialized():
                dist.barrier()

            # Step 5: If not in new group, nothing more to do for this rank.
            if not is_in_new_group:
                logger.info(f"[ElasticMegatronMixin rank={my_rank}] Rank removed from DP group, returning")
                return

            # Step 6: Restore state from CPU to GPU
            self._restore_state_from_cpu()

            # Step 7: Broadcast parameters from new_world_ranks[0] to all other
            #         ranks in the new group (using pre-created sync_group).
            self._sync_params_to_new_members(new_world_ranks, sync_group=sync_group)

            # Step 8: Final barrier among the new group members
            if dist.is_initialized():
                dist.barrier()

            logger.info(
                f"[ElasticMegatronMixin rank={my_rank}] "
                f"DP group rebuild complete, new dp_size="
                f"{len(new_world_ranks) // (tensor_model_parallel_size * pipeline_model_parallel_size)}"
            )

        except Exception as e:
            logger.exception(f"[ElasticMegatronMixin rank={my_rank}] Failed to rebuild DP group: {e}")
            raise

    def _capture_state_to_cpu(self) -> None:
        """Capture model and optimizer state to CPU memory."""
        my_rank = dist.get_rank()
        logger.info(f"[ElasticMegatronMixin rank={my_rank}] Capturing state to CPU...")

        # Capture model
        if hasattr(self, "module") and self.module is not None:
            self._model_snapshot = ModelStateSnapshot.from_model(self.module)

        # Capture optimizer
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self._optimizer_snapshot = OptimizerStateSnapshot.from_optimizer(self.optimizer)

        # Offload to CPU
        self._offload_to_cpu()

        torch.cuda.empty_cache()
        gc.collect()

    def _offload_to_cpu(self) -> None:
        """Move model and optimizer to CPU, freeing GPU memory."""
        my_rank = dist.get_rank()

        if self._model_on_gpu and hasattr(self, "module") and self.module is not None:
            logger.info(f"[ElasticMegatronMixin rank={my_rank}] Offloading model to CPU...")

            for module in self.module if isinstance(self.module, list) else [self.module]:
                unwrapped = module
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module

                for param in unwrapped.parameters():
                    param.data = param.data.cpu()
                    if param.grad is not None:
                        param.grad = param.grad.cpu()

            self._model_on_gpu = False

        if self._optimizer_on_gpu and hasattr(self, "optimizer") and self.optimizer is not None:
            logger.info(f"[ElasticMegatronMixin rank={my_rank}] Offloading optimizer to CPU...")

            if hasattr(self.optimizer, "offload_to_cpu"):
                self.optimizer.offload_to_cpu()
            else:
                # Manual offload for standard optimizers
                if hasattr(self.optimizer, "state"):
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cpu()

            self._optimizer_on_gpu = False

        torch.cuda.empty_cache()
        gc.collect()

    def _destroy_parallel_groups(self) -> None:
        """Destroy all Megatron parallel groups."""
        my_rank = dist.get_rank()
        logger.info(f"[ElasticMegatronMixin rank={my_rank}] Destroying parallel groups...")

        try:
            mpu.destroy_model_parallel()
        except Exception as e:
            logger.warning(f"[ElasticMegatronMixin rank={my_rank}] Error destroying parallel groups: {e}")

    def _reinitialize_parallel_groups(
        self,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
    ) -> None:
        """Reinitialize Megatron parallel groups with new configuration."""
        my_rank = dist.get_rank()

        logger.info(
            f"[ElasticMegatronMixin rank={my_rank}] "
            f"Reinitializing parallel groups: "
            f"tp={tensor_model_parallel_size}, "
            f"pp={pipeline_model_parallel_size}, "
            f"cp={context_parallel_size}"
        )

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=getattr(
                self.engine_config, "virtual_pipeline_model_parallel_size", None
            ),
            use_sharp=False,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )

    def _restore_state_from_cpu(self) -> None:
        """Restore model and optimizer state from CPU memory to GPU."""
        my_rank = dist.get_rank()
        logger.info(f"[ElasticMegatronMixin rank={my_rank}] Restoring state from CPU to GPU...")

        device = get_device_name()

        # Restore model
        if not self._model_on_gpu and self._model_snapshot is not None:
            if hasattr(self, "module") and self.module is not None:
                self._model_snapshot.to_model(self.module, device=device)
                self._model_on_gpu = True

        # Restore optimizer
        if not self._optimizer_on_gpu and self._optimizer_snapshot is not None:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self._optimizer_snapshot.to_optimizer(self.optimizer, device=device)
                self._optimizer_on_gpu = True

        torch.cuda.empty_cache()

    def _sync_params_to_new_members(
        self,
        new_world_ranks: list[int],
        sync_group=None,
    ) -> None:
        """Broadcast parameters from new_world_ranks[0] to all ranks in the group.

        ``sync_group`` must be pre-created by the caller (via ``dist.new_group``)
        **before** any early-return so that all global ranks participate in the
        collective.  Passing it in avoids a second ``dist.new_group`` call here.
        """
        my_rank = dist.get_rank()

        if not hasattr(self, "module") or self.module is None:
            return

        logger.info(
            f"[ElasticMegatronMixin rank={my_rank}] Syncing params via broadcast from rank {new_world_ranks[0]}"
        )

        src_rank = new_world_ranks[0]

        for module in self.module if isinstance(self.module, list) else [self.module]:
            unwrapped = module
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            for param in unwrapped.parameters():
                # Guard: NCCL only supports CUDA tensors.
                if param.data.device.type != "cuda":
                    param.data = param.data.cuda()
                dist.broadcast(param.data, src=src_rank, group=sync_group)


@dataclass
class ModelStateSnapshot:
    """Snapshot of model parameters and buffers stored in CPU memory."""

    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    num_parameters: int = 0
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_model(cls, model: list) -> "ModelStateSnapshot":
        """Capture model state from GPU to CPU memory.

        Args:
            model: List of model chunks (Megatron format)

        Returns:
            ModelStateSnapshot with parameters and buffers on CPU
        """
        snapshot = cls()

        # Handle both list of modules and single module
        if isinstance(model, list):
            models_to_process = model
        else:
            models_to_process = [model]

        for module in models_to_process:
            # Unwrap DDP/FSDP if present
            unwrapped = module
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            # Capture parameters
            for name, param in unwrapped.named_parameters():
                snapshot.state_dict[name] = param.data.detach().cpu().clone()
                snapshot.num_parameters += param.numel()
                snapshot.dtype = param.dtype

            # Capture buffers
            for name, buffer in unwrapped.named_buffers():
                snapshot.state_dict[f"{name}.buffer"] = buffer.data.detach().cpu().clone()

        if torch.distributed.get_rank() == 0:
            logger.info(f"Model snapshot captured: {snapshot.num_parameters} parameters, dtype={snapshot.dtype}")

        return snapshot

    def to_model(self, model: list, device: str = "cuda") -> None:
        """Restore model state from CPU memory to GPU.

        Args:
            model: List of model chunks to restore to
            device: Target device ("cuda" or "cpu")
        """
        if isinstance(model, list):
            models_to_process = model
        else:
            models_to_process = [model]

        with torch.no_grad():
            for module in models_to_process:
                unwrapped = module
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module

                for name, param in unwrapped.named_parameters():
                    if name in self.state_dict:
                        # Use direct assignment instead of copy_ so that params
                        # offloaded to CPU (via param.data = param.data.cpu())
                        # are properly moved back to the target device.
                        param.data = self.state_dict[name].to(device=device)

                for name, buffer in unwrapped.named_buffers():
                    full_name = f"{name}.buffer"
                    if full_name in self.state_dict:
                        buffer.data = self.state_dict[full_name].to(device=device)


@dataclass
class OptimizerStateSnapshot:
    """Snapshot of optimizer state stored in CPU memory."""

    state_dict: dict[str, Any] = field(default_factory=dict)
    param_groups_data: dict[str, Any] = field(default_factory=dict)
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_optimizer(cls, optimizer) -> "OptimizerStateSnapshot":
        """Capture optimizer state from GPU to CPU memory.

        Args:
            optimizer: MegatronOptimizer instance

        Returns:
            OptimizerStateSnapshot with state on CPU
        """
        snapshot = cls()

        if optimizer is None:
            return snapshot

        # Get base optimizer
        base_opt = optimizer
        while hasattr(base_opt, "optimizer"):
            base_opt = base_opt.optimizer

        snapshot.dtype = torch.float32

        # Capture optimizer state
        if hasattr(base_opt, "state"):
            for param_id, state in enumerate(base_opt.state.values()):
                state_key = f"param_{param_id}"
                snapshot.state_dict[state_key] = {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()
                }

        # Capture param groups
        if hasattr(base_opt, "param_groups"):
            for pg_id, pg in enumerate(base_opt.param_groups):
                pg_key = f"param_group_{pg_id}"
                snapshot.param_groups_data[pg_key] = {k: v for k, v in pg.items() if not isinstance(v, list | dict)}

        if torch.distributed.get_rank() == 0:
            logger.info(f"Optimizer snapshot captured: {len(snapshot.state_dict)} state dicts")

        return snapshot

    def to_optimizer(self, optimizer, device: str = "cuda") -> None:
        """Restore optimizer state from CPU memory to GPU.

        Args:
            optimizer: MegatronOptimizer to restore to
            device: Target device
        """
        if optimizer is None:
            return

        base_opt = optimizer
        while hasattr(base_opt, "optimizer"):
            base_opt = base_opt.optimizer

        # Restore optimizer state
        if hasattr(base_opt, "state"):
            for param_id, state in enumerate(base_opt.state.values()):
                state_key = f"param_{param_id}"
                if state_key in self.state_dict:
                    saved_state = self.state_dict[state_key]
                    for k, v in saved_state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device=device, non_blocking=True)
                        else:
                            state[k] = v


@EngineRegistry.register(model_type="language_model", backend="megatron_elastic")
class ElasticMegatronEngineWithLMHead(MegatronEngineWithLMHead, ElasticMegatronMixin):
    """
    Elastic Megatron Engine with LM head for language modeling.

    Combines MegatronEngineWithLMHead (full forward/backward logic)
    with ElasticMegatronMixin (DP rebuild capability).
    """

    pass


@EngineRegistry.register(model_type="value_model", backend="megatron_elastic")
class ElasticMegatronEngineWithValueHead(MegatronEngineWithValueHead, ElasticMegatronMixin):
    """
    Elastic Megatron Engine with value head for value modeling.

    Combines MegatronEngineWithValueHead (full forward/backward logic)
    with ElasticMegatronMixin (DP rebuild capability).
    """

    pass
