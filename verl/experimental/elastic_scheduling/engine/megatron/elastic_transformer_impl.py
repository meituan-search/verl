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
Elastic Megatron Mixin for dynamic DP group resizing.

Provides:
- ElasticMegatronMixin: Mixin that adds DP rebuild capability to any Megatron engine

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

from verl.utils.device import get_device_name

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
        Replace the Megatron DP process group without touching TP/PP/CP groups.

        Key insight: ``dist.get_world_size()`` always returns the full world size
        (e.g. 6) even after ``destroy_model_parallel()``.  Calling
        ``initialize_model_parallel`` afterwards therefore always builds a DP
        group of size 6 — it cannot produce a 5-rank DP group from a 6-rank
        world.  The old destroy+reinit approach was fundamentally broken.

        Correct approach — only replace the DP communicator:
        1. ``dist.new_group(new_world_ranks)`` — one collective, ALL current
           ranks must call it simultaneously. Removed ranks participate and then
           return early; retained ranks get the new group handle.
        2. Patch the relevant ``mpu`` module-level globals so that
           ``get_data_parallel_group/rank/world_size`` all reflect the new DP
           membership.  TP/PP/CP groups are left completely untouched.
        3. (Retained ranks only) restore model state to GPU, broadcast params
           to any newly added ranks via the new group.

        Args:
            new_world_ranks: Sorted list of global ranks in the new DP group.
            tensor_model_parallel_size: Unused — kept for API compat.
            pipeline_model_parallel_size: Unused — kept for API compat.
            context_parallel_size: Unused — kept for API compat.
            expert_model_parallel_size: Unused — kept for API compat.
            expert_tensor_parallel_size: Unused — kept for API compat.
        """
        my_rank = dist.get_rank()

        # Lazy-init mixin state when the class was injected via __class__
        # assignment (no __init__ call was made).
        if not hasattr(self, "_model_on_gpu"):
            self._model_on_gpu = True
            self._optimizer_on_gpu = True
            self._model_snapshot = None
            self._optimizer_snapshot = None

        is_in_new_group = my_rank in new_world_ranks
        new_dp_size = len(new_world_ranks)

        logger.info(
            f"[ElasticMegatronMixin rank={my_rank}] rebuild_dp_group called\n"
            f"  new_world_ranks={new_world_ranks}\n"
            f"  is_in_new_group={is_in_new_group}"
        )

        try:
            # Step 1: Only retained ranks need a CPU snapshot + restore cycle.
            # Removed ranks will have their weights offloaded by switch_to_rollout()
            # right after this call, so capturing state here would be redundant.
            if is_in_new_group:
                self._capture_state_to_cpu()

            # Step 2: Create the new DP process group.
            # dist.new_group() is a collective — every rank in the current world
            # must call it simultaneously, including ranks being removed.
            new_dp_group = dist.new_group(ranks=new_world_ranks)

            # Step 3: Global barrier so all ranks have finished the collective
            # before anyone proceeds.
            dist.barrier()

            # Step 4: Ranks NOT in the new group have fulfilled their collective
            # obligations and can exit now.
            if not is_in_new_group:
                logger.info(f"[ElasticMegatronMixin rank={my_rank}] Rank removed from DP group, returning")
                return

            # Step 5: Patch mpu so all Megatron code sees the new DP group.
            self._patch_mpu_dp_group(new_dp_group, new_world_ranks)

            # Step 6: Restore model + optimizer back to GPU.
            self._restore_state_from_cpu()

            # Step 7: Broadcast params to any newly joined ranks.
            prev_ranks = getattr(self, "_prev_dp_world_ranks", set(new_world_ranks))
            has_new_members = any(r not in prev_ranks for r in new_world_ranks)
            if has_new_members:
                self._sync_params_to_new_members(new_world_ranks, sync_group=new_dp_group)
            self._prev_dp_world_ranks = set(new_world_ranks)

            # Step 8: Final barrier within the new group.
            dist.barrier(group=new_dp_group)

            logger.info(f"[ElasticMegatronMixin rank={my_rank}] DP group rebuild complete, new dp_size={new_dp_size}")

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
                try:
                    self.optimizer.offload_to_cpu()
                except NameError:
                    # Megatron's offload_to_cpu has a bug: `logging` is not
                    # imported in megatron/core/optimizer/optimizer.py, causing
                    # NameError inside log_single_rank(). Fall back to manual
                    # offload which achieves the same effect.
                    self._manual_offload_optimizer()
            else:
                self._manual_offload_optimizer()

            self._optimizer_on_gpu = False

        torch.cuda.empty_cache()
        gc.collect()

    def _manual_offload_optimizer(self) -> None:
        """Move optimizer state tensors to CPU manually.

        Used as a fallback when ``optimizer.offload_to_cpu()`` is unavailable
        or broken (e.g. Megatron's NameError bug on some versions).
        Walks the Megatron optimizer wrapper chain to reach the base optimizer
        and moves every state tensor to CPU in-place.
        """
        opt = self.optimizer
        # Unwrap Megatron optimizer wrappers to reach the base PyTorch optimizer.
        while hasattr(opt, "optimizer"):
            opt = opt.optimizer
        if hasattr(opt, "state"):
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

    def _patch_mpu_dp_group(self, new_dp_group, new_world_ranks: list[int]) -> None:
        """
        Patch Megatron's module-level DP group globals so that all subsequent
        calls to ``mpu.get_data_parallel_group/rank/world_size`` reflect the
        new DP membership.

        TP / PP / CP groups are left completely untouched — only the four DP
        globals that matter for gradient all-reduce and DP-size queries are
        replaced.

        ``mpu.get_data_parallel_world_size()`` checks ``_MPU_DATA_PARALLEL_WORLD_SIZE``
        first (non-None overrides the group query), so we set it explicitly.
        Similarly ``_MPU_DATA_PARALLEL_RANK`` overrides the rank query.
        """
        import megatron.core.parallel_state as _mpu_mod

        my_rank = dist.get_rank()
        new_dp_size = len(new_world_ranks)
        new_dp_rank = new_world_ranks.index(my_rank)

        # Replace the primary DP group handle.
        _mpu_mod._DATA_PARALLEL_GROUP = new_dp_group
        # _DATA_PARALLEL_GROUP_WITH_CP is used when CP is enabled; for CP=1
        # it is the same group.  Safe to alias unconditionally.
        _mpu_mod._DATA_PARALLEL_GROUP_WITH_CP = new_dp_group

        # Override the "on-the-fly" size / rank values so they take priority
        # over the group-based queries in get_data_parallel_world_size/rank.
        _mpu_mod._MPU_DATA_PARALLEL_WORLD_SIZE = new_dp_size
        _mpu_mod._MPU_DATA_PARALLEL_RANK = new_dp_rank

        # Update the global ranks list used by checkpoint / broadcast helpers.
        _mpu_mod._DATA_PARALLEL_GLOBAL_RANKS = new_world_ranks
        _mpu_mod._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = new_world_ranks

        logger.info(
            f"[ElasticMegatronMixin rank={my_rank}] mpu DP group patched: "
            f"dp_size={new_dp_size}, dp_rank={new_dp_rank}, ranks={new_world_ranks}"
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
