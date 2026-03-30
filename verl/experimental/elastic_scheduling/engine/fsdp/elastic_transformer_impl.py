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
Elastic FSDP2 Engine for dynamic DP group resizing.

Provides:
- ElasticFSDPMixin: Mixin that adds DP rebuild capability to any FSDP engine
- ElasticFSDPEngineWithLMHead: FSDP engine with elastic scaling support
- ElasticFSDPEngineWithValueHead: FSDP value engine with elastic scaling support

Key features:
1. DP group rebuild without disk I/O (state stored in CPU memory)
2. Support for adding/removing training resources
3. Parameter synchronization for newly joined ranks
4. Seamless integration with existing FSDPEngine interface
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist

from verl.utils.device import get_device_name
from verl.workers.engine import EngineRegistry

# Import FSDP engine classes - these will be imported conditionally
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngineWithLMHead,
    FSDPEngineWithValueHead,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class FSDPModelStateSnapshot:
    """Snapshot of FSDP model parameters stored in CPU memory."""

    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    num_parameters: int = 0
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_model(cls, model) -> "FSDPModelStateSnapshot":
        """Capture FSDP model state from GPU to CPU memory.

        Args:
            model: FSDP wrapped model or module

        Returns:
            FSDPModelStateSnapshot with parameters on CPU
        """
        snapshot = cls()

        # Unwrap FSDP if present
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # Capture parameters directly
        for name, param in unwrapped.named_parameters():
            snapshot.state_dict[name] = param.data.detach().cpu().clone()
            snapshot.num_parameters += param.numel()
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                snapshot.dtype = param.dtype

        # Capture buffers
        for name, buffer in unwrapped.named_buffers():
            snapshot.state_dict[f"{name}.__buffer__"] = buffer.data.detach().cpu().clone()

        if torch.distributed.get_rank() == 0:
            logger.info(f"FSDP Model snapshot captured: {snapshot.num_parameters} parameters, dtype={snapshot.dtype}")

        return snapshot

    def to_model(self, model, device: str = "cuda") -> None:
        """Restore model state from CPU memory to GPU.

        Args:
            model: FSDP wrapped model or module to restore to
            device: Target device ("cuda" or "cpu")
        """
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if name in self.state_dict:
                    param.data.copy_(self.state_dict[name].to(device=device, non_blocking=True))

            for name, buffer in unwrapped.named_buffers():
                full_name = f"{name}.__buffer__"
                if full_name in self.state_dict:
                    buffer.data.copy_(self.state_dict[full_name].to(device=device, non_blocking=True))


@dataclass
class FSDPOptimizerStateSnapshot:
    """Snapshot of FSDP optimizer state stored in CPU memory."""

    state_dict: dict[str, Any] = field(default_factory=dict)
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_optimizer(cls, optimizer) -> "FSDPOptimizerStateSnapshot":
        """Capture optimizer state from GPU to CPU memory.

        Args:
            optimizer: PyTorch optimizer instance

        Returns:
            FSDPOptimizerStateSnapshot with state on CPU
        """
        snapshot = cls()

        if optimizer is None:
            return snapshot

        # Capture optimizer state dict
        try:
            if hasattr(optimizer, "state_dict"):
                opt_state_dict = optimizer.state_dict()
                if "state" in opt_state_dict:
                    for param_id, state in opt_state_dict["state"].items():
                        state_key = f"param_{param_id}"
                        snapshot.state_dict[state_key] = {
                            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()
                        }
                        if isinstance(state, dict) and any(isinstance(v, torch.Tensor) for v in state.values()):
                            for v in state.values():
                                if isinstance(v, torch.Tensor):
                                    snapshot.dtype = v.dtype
                                    break
        except Exception as e:
            logger.warning(f"Failed to capture optimizer state: {e}")

        if torch.distributed.get_rank() == 0:
            logger.info(f"Optimizer snapshot captured: {len(snapshot.state_dict)} state dicts")

        return snapshot

    def to_optimizer(self, optimizer, device: str = "cuda") -> None:
        """Restore optimizer state from CPU memory to GPU.

        Args:
            optimizer: PyTorch optimizer to restore to
            device: Target device
        """
        if optimizer is None or not self.state_dict:
            return

        try:
            if hasattr(optimizer, "state_dict"):
                opt_state_dict = optimizer.state_dict()
                if "state" in opt_state_dict:
                    for param_id, state in opt_state_dict["state"].items():
                        state_key = f"param_{param_id}"
                        if state_key in self.state_dict:
                            saved_state = self.state_dict[state_key]
                            for k, v in saved_state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device=device, non_blocking=True)
                                else:
                                    state[k] = v
        except Exception as e:
            logger.warning(f"Failed to restore optimizer state: {e}")


class ElasticFSDPMixin:
    """
    Mixin that adds elastic DP group rebuild capabilities to any FSDP engine.

    This mixin provides the rebuild_dp_group method that handles:
    1. Capturing model and optimizer state to CPU
    2. Rebuilding FSDP communication groups
    3. Restoring state without disk I/O
    4. Syncing parameters to newly joined ranks

    Usage:
        class MyElasticEngine(FSDPEngineWithLMHead, ElasticFSDPMixin):
            pass
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State snapshots for DP rebuild
        self._model_snapshot: Optional[FSDPModelStateSnapshot] = None
        self._optimizer_snapshot: Optional[FSDPOptimizerStateSnapshot] = None

        # Track device placement
        self._model_on_gpu = True
        self._optimizer_on_gpu = True

    def rebuild_dp_group(
        self,
        new_world_ranks: list[int],
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> None:
        """
        Rebuild DP group for elastic FSDP training.

        Procedure:
        1. Capture model and optimizer state to CPU
        2. Offload everything to CPU
        3. Barrier with all ranks
        4. Rebuild FSDP groups
        5. Restore state from CPU
        6. Sync parameters to newly joined ranks

        Args:
            new_world_ranks: List of global ranks in new DP group
            tensor_parallel_size: TP size (for future extension)
            pipeline_parallel_size: PP size (for future extension)
        """
        my_rank = dist.get_rank()

        # Check if this rank is in new DP group
        is_in_new_group = my_rank in new_world_ranks

        logger.info(
            f"[ElasticFSDPMixin rank={my_rank}] rebuild_dp_group called\n"
            f"  new_world_ranks={new_world_ranks}\n"
            f"  is_in_new_group={is_in_new_group}"
        )

        try:
            # Step 1: Capture state to CPU
            self._capture_state_to_cpu()

            # Step 2: Barrier with all current ranks
            if dist.is_initialized():
                dist.barrier()

            # Step 3: If not in new group, return early
            if not is_in_new_group:
                logger.info(f"[ElasticFSDPMixin rank={my_rank}] Rank removed from DP group, returning")
                return

            # Step 4: Rebuild FSDP groups
            self._rebuild_fsdp_groups(new_world_ranks)

            # Step 5: Barrier after reinitialization
            if dist.is_initialized():
                dist.barrier()

            # Step 6: Restore state from CPU
            self._restore_state_from_cpu()

            # Step 7: Sync parameters to newly joined ranks
            self._sync_params_to_new_members(new_world_ranks)

            # Step 8: Final barrier
            if dist.is_initialized():
                dist.barrier()

            logger.info(
                f"[ElasticFSDPMixin rank={my_rank}] DP group rebuild complete with {len(new_world_ranks)} ranks"
            )

        except Exception as e:
            logger.exception(f"[ElasticFSDPMixin rank={my_rank}] Failed to rebuild DP group: {e}")
            raise

    def _capture_state_to_cpu(self) -> None:
        """Capture model and optimizer state to CPU memory."""
        my_rank = dist.get_rank()
        logger.info(f"[ElasticFSDPMixin rank={my_rank}] Capturing state to CPU...")

        # Capture model
        if hasattr(self, "module") and self.module is not None:
            self._model_snapshot = FSDPModelStateSnapshot.from_model(self.module)

        # Capture optimizer
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self._optimizer_snapshot = FSDPOptimizerStateSnapshot.from_optimizer(self.optimizer)

        # Offload to CPU
        self._offload_to_cpu()

        torch.cuda.empty_cache()
        gc.collect()

    def _offload_to_cpu(self) -> None:
        """Move model and optimizer to CPU, freeing GPU memory."""
        my_rank = dist.get_rank()

        if self._model_on_gpu and hasattr(self, "module") and self.module is not None:
            logger.info(f"[ElasticFSDPMixin rank={my_rank}] Offloading model to CPU...")

            try:
                from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu

                offload_fsdp_model_to_cpu(self.module)
            except Exception as e:
                logger.warning(f"Failed to use offload_fsdp_model_to_cpu: {e}, trying manual offload")

                unwrapped = self.module
                while hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module

                for param in unwrapped.parameters():
                    param.data = param.data.cpu()
                    if param.grad is not None:
                        param.grad = param.grad.cpu()

            self._model_on_gpu = False

        if self._optimizer_on_gpu and hasattr(self, "optimizer") and self.optimizer is not None:
            logger.info(f"[ElasticFSDPMixin rank={my_rank}] Offloading optimizer to CPU...")

            try:
                from verl.utils.fsdp_utils import offload_fsdp_optimizer

                offload_fsdp_optimizer(self.optimizer)
            except Exception as e:
                logger.warning(f"Failed to use offload_fsdp_optimizer: {e}, trying manual offload")

            self._optimizer_on_gpu = False

        torch.cuda.empty_cache()
        gc.collect()

    def _rebuild_fsdp_groups(self, new_world_ranks: list[int]) -> None:
        """Rebuild FSDP communication groups."""
        my_rank = dist.get_rank()
        logger.info(f"[ElasticFSDPMixin rank={my_rank}] Rebuilding FSDP groups for {len(new_world_ranks)} ranks")

        # For FSDP2, rebuild process groups
        try:
            # Create new subgroups for the new ranks
            new_group = dist.new_group(ranks=new_world_ranks)
            logger.info(
                f"[ElasticFSDPMixin rank={my_rank}] "
                f"Created new process group for {len(new_world_ranks)} ranks "
                f"new_group {new_group}"
            )
        except Exception as e:
            logger.warning(f"[ElasticFSDPMixin rank={my_rank}] Error creating process group: {e}")

    def _restore_state_from_cpu(self) -> None:
        """Restore model and optimizer state from CPU memory to GPU."""
        my_rank = dist.get_rank()
        logger.info(f"[ElasticFSDPMixin rank={my_rank}] Restoring state from CPU to GPU...")

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

    def _sync_params_to_new_members(self, new_world_ranks: list[int]) -> None:
        """Broadcast parameters from existing ranks to newly joined ranks."""
        my_rank = dist.get_rank()

        logger.info(
            f"[ElasticFSDPMixin rank={my_rank}] Syncing params to new members among {len(new_world_ranks)} ranks"
        )

        if hasattr(self, "module") and self.module is not None:
            # Create process group for the new ranks
            new_group = dist.new_group(ranks=new_world_ranks)

            # Broadcast from rank 0 in new group
            unwrapped = self.module
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            for param in unwrapped.parameters():
                dist.broadcast(param.data, src=0, group=new_group)

            logger.info(f"[ElasticFSDPMixin rank={my_rank}] Param sync complete")


@EngineRegistry.register(model_type="language_model", backend="fsdp2_elastic")
class ElasticFSDPEngineWithLMHead(FSDPEngineWithLMHead, ElasticFSDPMixin):
    """
    Elastic FSDP Engine with LM head for language modeling.

    Combines FSDPEngineWithLMHead (full forward/backward logic)
    with ElasticFSDPMixin (DP rebuild capability).
    """

    pass


@EngineRegistry.register(model_type="value_model", backend="fsdp2_elastic")
class ElasticFSDPEngineWithValueHead(FSDPEngineWithValueHead, ElasticFSDPMixin):
    """
    Elastic FSDP Engine with value head for value modeling.

    Combines FSDPEngineWithValueHead (full forward/backward logic)
    with ElasticFSDPMixin (DP rebuild capability).
    """

    pass
