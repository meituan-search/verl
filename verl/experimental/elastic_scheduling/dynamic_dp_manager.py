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
Dynamic Data Parallel Manager for Megatron-LM

This module enables dynamic resizing of Data Parallel groups during training,
allowing resources to be added/removed without checkpointing to disk.
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.optimizer import MegatronOptimizer
from torch import Tensor


@dataclass
class ModelStateSnapshot:
    """Snapshot of model parameters stored in CPU memory."""

    state_dict: dict[str, Tensor] = field(default_factory=dict)
    num_parameters: int = 0
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_model(cls, model) -> "ModelStateSnapshot":
        """Capture model state from GPU to CPU memory."""
        snapshot = cls()
        unwrapped = model[0].module if hasattr(model[0], "module") else model[0]
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        for name, param in unwrapped.named_parameters():
            snapshot.state_dict[name] = param.data.detach().cpu().clone()
            snapshot.num_parameters += param.numel()
            snapshot.dtype = param.dtype

        # Also capture buffer states
        for name, buffer in unwrapped.named_buffers():
            snapshot.state_dict[f"{name}.buffer"] = buffer.data.detach().cpu().clone()

        return snapshot

    def to_model(self, model, device: str = "cuda") -> None:
        """Restore model state from CPU memory to GPU."""
        unwrapped = model[0].module if hasattr(model[0], "module") else model[0]
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if name in self.state_dict:
                    param.data.copy_(self.state_dict[name].to(device=device, non_blocking=True))

            for name, buffer in unwrapped.named_buffers():
                full_name = f"{name}.buffer"
                if full_name in self.state_dict:
                    buffer.data.copy_(self.state_dict[full_name].to(device=device, non_blocking=True))


@dataclass
class OptimizerStateSnapshot:
    """Snapshot of optimizer state stored in CPU memory."""

    state_dict: dict[str, Any] = field(default_factory=dict)
    param_groups_shapes: dict[int, list[int]] = field(default_factory=dict)
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_optimizer(cls, optimizer: MegatronOptimizer) -> "OptimizerStateSnapshot":
        """Capture optimizer state from GPU to CPU memory."""
        snapshot = cls()

        if optimizer is None:
            return snapshot

        # Get base optimizer
        base_opt = optimizer
        while hasattr(base_opt, "optimizer"):
            base_opt = base_opt.optimizer

        snapshot.dtype = torch.float32
        if hasattr(base_opt, "param_groups") and base_opt.param_groups:
            for i, pg in enumerate(base_opt.param_groups):
                for j, p in enumerate(pg.get("params", [])):
                    # Store optimizer state
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        snapshot.state_dict[f"pg{i}_p{j}"] = {
                            k: v.detach().cpu().clone() if isinstance(v, Tensor) else v for k, v in state.items()
                        }
                    # Store param shape info
                    snapshot.param_groups_shapes[f"pg{i}_p{j}"] = list(p.shape)
                    snapshot.dtype = (
                        p.dtype if p.dtype in [torch.float32, torch.float16, torch.bfloat16] else torch.float32
                    )

        return snapshot

    def to_optimizer(self, optimizer: MegatronOptimizer, model, device: str = "cuda") -> None:
        """Restore optimizer state from CPU memory to GPU."""
        if optimizer is None:
            return

        # Get base optimizer
        base_opt = optimizer
        while hasattr(base_opt, "optimizer"):
            base_opt = base_opt.optimizer

        unwrapped = model[0].module if hasattr(model[0], "module") else model[0]
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # Build param mapping
        param_map = {}
        for i, pg in enumerate(base_opt.param_groups):
            for j, p in enumerate(pg.get("params", [])):
                key = f"pg{i}_p{j}"
                if key in self.state_dict:
                    param_map[p] = self.state_dict[key]

        # Restore state
        for p, state in optimizer.state.items():
            key = None
            for i, pg in enumerate(base_opt.param_groups):
                for j, pg_p in enumerate(pg.get("params", [])):
                    if pg_p is p or (hasattr(pg_p, "data") and pg_p.data is p):
                        key = f"pg{i}_p{j}"
                        break
                if key:
                    break

            if key and key in self.state_dict:
                saved_state = self.state_dict[key]
                for k, v in saved_state.items():
                    if isinstance(v, Tensor):
                        state[k] = v.to(device=device, non_blocking=True)
                    else:
                        state[k] = v


class DynamicDPManager:
    """
    Manager for dynamically rebuilding Data Parallel groups.

    This enables:
    - Adding/removing DP resources during training
    - Rebuilding DP groups without checkpointing to disk
    - Restoring model and optimizer state from CPU memory
    """

    def __init__(
        self,
        model,
        optimizer: Optional[MegatronOptimizer] = None,
        use_cpu_offload: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_cpu_offload = use_cpu_offload

        # Current state snapshots
        self.model_snapshot: Optional[ModelStateSnapshot] = None
        self.optimizer_snapshot: Optional[OptimizerStateSnapshot] = None

        # Track original DP configuration
        self.original_dp_size = parallel_state.get_data_parallel_world_size()
        self.original_dp_rank = parallel_state.get_data_parallel_rank()
        self.original_world_size = dist.get_world_size()

        # Device management
        self.model_on_gpu = True
        self.optimizer_on_gpu = True

    def capture_state(self) -> tuple[ModelStateSnapshot, OptimizerStateSnapshot]:
        """
        Capture current model and optimizer state to CPU memory.
        This should be called before destroying the current DP group.
        """
        print(f"[Rank {dist.get_rank()}] Capturing state to CPU memory...")

        # Capture model state
        self.model_snapshot = ModelStateSnapshot.from_model(self.model)

        # Capture optimizer state
        self.optimizer_snapshot = OptimizerStateSnapshot.from_optimizer(self.optimizer)

        # Clear CUDA cache
        if self.use_cpu_offload:
            torch.cuda.empty_cache()
            gc.collect()

        return self.model_snapshot, self.optimizer_snapshot

    def offload_to_cpu(self):
        """Offload model and optimizer to CPU, clearing GPU memory."""
        if self.model_on_gpu:
            print(f"[Rank {dist.get_rank()}] Offloading model to CPU...")
            # Move parameters to CPU
            unwrapped = self.model[0].module if hasattr(self.model[0], "module") else self.model[0]
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            for param in unwrapped.parameters():
                param.data = param.data.cpu()
                if param.grad is not None:
                    param.grad = param.grad.cpu()

            self.model_on_gpu = False

        if self.optimizer_on_gpu and self.optimizer is not None:
            print(f"[Rank {dist.get_rank()}] Offloading optimizer to CPU...")
            self.optimizer.offload_to_cpu()
            self.optimizer_on_gpu = False

        torch.cuda.empty_cache()
        gc.collect()

    def load_to_gpu(self):
        """Load model and optimizer back to GPU."""
        if not self.model_on_gpu:
            print(f"[Rank {dist.get_rank()}] Loading model to GPU...")
            unwrapped = self.model[0].module if hasattr(self.model[0], "module") else self.model[0]
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            for param in unwrapped.parameters():
                param.data = param.data.cuda()

            self.model_on_gpu = True

        if not self.optimizer_on_gpu and self.optimizer is not None:
            print(f"[Rank {dist.get_rank()}] Loading optimizer to GPU...")
            self.optimizer.load_from_cpu()
            self.optimizer_on_gpu = True

        torch.cuda.empty_cache()

    def restore_state(
        self, model_state: Optional[ModelStateSnapshot] = None, optimizer_state: Optional[OptimizerStateSnapshot] = None
    ):
        """
        Restore model and optimizer state from CPU memory snapshots.
        """
        model_state = model_state or self.model_snapshot
        optimizer_state = optimizer_state or self.optimizer_snapshot

        if model_state is not None:
            print(f"[Rank {dist.get_rank()}] Restoring model state...")
            self.model_snapshot.to_model(self.model, device="cuda")

        if optimizer_state is not None:
            print(f"[Rank {dist.get_rank()}] Restoring optimizer state...")
            self.optimizer_snapshot.to_optimizer(self.optimizer, self.model, device="cuda")

    def destroy_parallel_groups(self):
        """
        Destroy all Megatron parallel groups.
        This must be called when changing world_size.
        """
        print(f"[Rank {dist.get_rank()}] Destroying parallel groups...")
        parallel_state.destroy_model_parallel()

    @contextmanager
    def rebuild_dp_group(self, new_dp_size: int):
        """
        Context manager for rebuilding DP groups.

        Usage:
            with manager.rebuild_dp_group(new_dp_size=4):
                # Inside this context, DP groups are rebuilt
                # Training continues with new configuration

        Args:
            new_dp_size: New data parallel size after rebuild

        Note:
            This assumes torchrun has already updated the world_size.
            For adding resources, you would restart with more processes.
        """
        # Step 1: Capture current state to CPU
        self.capture_state()

        # Step 2: Offload everything to CPU
        self.offload_to_cpu()

        # Step 3: Destroy old parallel groups
        self.destroy_parallel_groups()

        # Step 4: Barrier to synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

        try:
            # Step 5: Recalculate parallel configuration
            world_size = dist.get_world_size()
            args = self._get_args()

            # Assuming TP and PP remain the same
            tp_size = args.tensor_model_parallel_size
            pp_size = args.pipeline_model_parallel_size

            # Validate new configuration
            assert world_size % (tp_size * pp_size) == 0, (
                f"world_size {world_size} not divisible by TP*PP ({tp_size}*{pp_size})"
            )

            new_actual_dp_size = world_size // (tp_size * pp_size)

            print(f"[Rank {dist.get_rank()}] Rebuilding with DP={new_actual_dp_size} (requested {new_dp_size})")

            # Step 6: Reinitialize parallel groups
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
                # Other parameters should be passed from original config
            )

            # Step 7: Barrier after reinitialization
            dist.barrier()

            # Step 8: Restore state
            self.load_to_gpu()
            self.restore_state()

            # Step 9: Rebuild DDP wrapper if needed
            self._rebuild_ddp()

            print(f"[Rank {dist.get_rank()}] DP group rebuild complete")

            yield new_actual_dp_size

        except Exception as e:
            print(f"[Rank {dist.get_rank()}] Error during DP rebuild: {e}")
            raise

        finally:
            # Cleanup if needed
            pass

    def _get_args(self):
        """Get Megatron arguments."""
        from megatron.training.global_vars import get_args

        return get_args()

    def _rebuild_ddp(self):
        """Rebuild DDP wrapper after parallel group changes."""
        # This would need to be customized based on the DDP implementation
        # For Custom FSDP, you would need to reinitialize the wrapper
        pass


class FlexibleDPManager(DynamicDPManager):
    """
    Extended DynamicDPManager with support for:
    - Splitting a DP group into multiple subgroups
    - Merging multiple DP groups
    - Dynamic resource allocation between train and rollout
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Custom subgroups for partial DP sync
        self.train_subgroup: Optional[dist.ProcessGroup] = None
        self.rollout_subgroup: Optional[dist.ProcessGroup] = None

        # Current mode: 'train', 'rollout', 'mixed'
        self.current_mode = "train"

    def create_partial_dp_groups(self, train_ranks: list[int]):
        """
        Create a subgroup for training ranks.

        Args:
            train_ranks: List of ranks that will participate in training
        """
        assert all(0 <= r < self.original_world_size for r in train_ranks), "Invalid rank in train_ranks"

        # Create subgroup for training ranks
        self.train_subgroup, _ = dist.new_subgroups_by_ranks(train_ranks)

        # All other ranks can be used for rollout
        self.rollout_ranks = [r for r in range(self.original_world_size) if r not in train_ranks]
        if self.rollout_ranks:
            self.rollout_subgroup, _ = dist.new_subgroups_by_ranks(self.rollout_ranks)

        print(f"[Rank {dist.get_rank()}] Created partial DP groups: train={train_ranks}, rollout={self.rollout_ranks}")

    def sync_gradients_partial(self):
        """Synchronize gradients only within the train subgroup."""
        if self.train_subgroup is None:
            return

        rank = dist.get_rank()
        if rank not in self.train_subgroup.ranks:
            return

        # Manual gradient sync for partial group
        unwrapped = self.model[0].module if hasattr(self.model[0], "module") else self.model[0]
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        for param in unwrapped.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.train_subgroup)
                param.grad /= len(self.train_subgroup.ranks)

    @contextmanager
    def switch_mode(self, mode: str):
        """
        Switch between training and rollout modes.

        Args:
            mode: 'train' for full DP sync, 'rollout' for inference only
        """
        old_mode = self.current_mode
        self.current_mode = mode

        try:
            if mode == "rollout":
                # Disable gradient sync
                ddp = self.model[0]
                if hasattr(ddp, "no_sync"):
                    self._rollout_context = ddp.no_sync()
                    self._rollout_context.__enter__()

            yield mode

        finally:
            if old_mode == "rollout":
                # Re-enable gradient sync
                self._rollout_context.__exit__(None, None, None)

            self.current_mode = old_mode


def example_usage():
    """
    Example usage of DynamicDPManager for RL training.
    """
    import torch.distributed as dist

    # Initialize
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(...)

    # Setup model and optimizer
    model = ...  # Your model
    optimizer = ...  # Your optimizer

    # Create dynamic manager
    manager = DynamicDPManager(model, optimizer)

    # Training loop with dynamic DP
    for iteration in range(1000):
        if iteration % 100 == 0:
            # Rebuild DP groups periodically (e.g., for elastic training)
            with manager.rebuild_dp_group(new_dp_size=2):
                pass  # train_step(model, data)
        else:
            pass  # train_step(model, data)


# For elastic training with torchrun
class ElasticDPManager(FlexibleDPManager):
    """
    Manager for elastic training with dynamic resource adjustment.

    Works with torchrun's elastic launch to handle adding/removing nodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_snapshot_buffer = {}

    def prepare_for_elastic_resize(self):
        """
        Prepare manager for elastic resize event.
        Called before torchrun resizes the world.
        """
        # Capture current state
        self.capture_state()
        self.offload_to_cpu()

        # Clear non-essential references
        self.model = None
        self.optimizer = None

        return self.model_snapshot, self.optimizer_snapshot

    def restore_from_elastic_resize(
        self, model_snapshot: ModelStateSnapshot, optimizer_snapshot: OptimizerStateSnapshot, model, optimizer
    ):
        """
        Restore from elastic resize event.
        Called after torchrun has resized the world.
        """
        # Reinitialize parallel groups with new world size
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(...)

        # Restore references
        self.model = model
        self.optimizer = optimizer

        # Load state back to GPU
        self.model_snapshot = model_snapshot
        self.optimizer_snapshot = optimizer_snapshot
        self.load_to_gpu()
        self.restore_state()
