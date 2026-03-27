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
Example usage of DynamicDPManager for Megatron-LM.

This demonstrates how to:
1. Dynamically resize DP groups without checkpointing to disk
2. Add/remove resources during training
3. Switch between train and rollout modes

Usage:
    # Single node (4 GPUs)
    torchrun --standalone --nnodes=1 --nproc_per_node=4 dynamic_dp_usage.py

    # Elastic training (dynamic node count)
    torchrun --standalone --nnodes=1:4 --nproc_per_node=4 dynamic_dp_usage.py --elastic
"""

import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.dynamic_dp_manager import (
    DynamicDPManager,
    ModelStateSnapshot,
)


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstration."""

    def __init__(self, vocab_size=32000, hidden_size=512, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=2048, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        return self.lm_head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic DP Manager Example")
    parser.add_argument("--elastic", action="store_true", help="Enable elastic training mode")
    parser.add_argument("--target-dp-size", type=int, default=2, help="Target DP size for rebuild")
    parser.add_argument("--sync-interval", type=int, default=100, help="Interval for DP sync")
    parser.add_argument(
        "--local-snapshot", action="store_true", default=True, help="Use local memory for state snapshots"
    )
    return parser.parse_args()


class MegatronSimulator:
    """
    Simulates Megatron-LM training loop with dynamic DP support.

    In a real scenario, you would replace this with actual Megatron training.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        model: nn.Module,
        dynamic_manager: Optional[DynamicDPManager] = None,
    ):
        self.world_size = world_size
        self.rank = rank
        self.model = model
        self.dynamic_manager = dynamic_manager
        self.iteration = 0

        # Simulate DP group
        self.dp_size = world_size  # Initially DP = world_size
        self.dp_rank = rank

    def train_step(self, batch_size=8, seq_len=128):
        """Simulate a training step."""
        # Generate dummy data
        local_rank = dist.get_rank() % torch.cuda.device_count()
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=f"cuda:{local_rank}")
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)

        # Forward pass
        if self.dynamic_manager and hasattr(self.dynamic_manager.model[0], "module"):
            # Use the model from dynamic manager if available
            model = self.dynamic_manager.model[0].module
        else:
            model = self.model

        logits = model(input_ids, attention_mask)

        # Simulate loss
        loss = logits.mean()

        # Backward pass
        if self.dynamic_manager and self.dynamic_manager.model_on_gpu:
            loss.backward()

        return loss.item()

    def rebuild_dp(self, new_dp_size: int):
        """
        Simulate DP group rebuild.

        In real Megatron, this would:
        1. Capture state to CPU
        2. Destroy parallel groups
        3. Reinitialize with new world_size
        4. Restore from CPU memory
        """
        print(f"[Rank {self.rank}] Rebuilding DP from {self.dp_size} to {new_dp_size}")

        if self.dynamic_manager:
            with self.dynamic_manager.rebuild_dp_group(new_dp_size):
                # Inside this context, the new DP group is active
                self.dp_size = new_dp_size
                self.dp_rank = self.rank % new_dp_size

                # Simulate some training
                self.train_step()
        else:
            # Without dynamic manager, just update local state
            self.dp_size = new_dp_size
            self.dp_rank = self.rank % new_dp_size

        print(f"[Rank {self.rank}] DP rebuild complete. DP_size={self.dp_size}, DP_rank={self.dp_rank}")


def setup_megatron_simulator(rank, world_size, args):
    """Setup simulator with Megatron-like initialization."""

    # Initialize distributed
    torch.cuda.set_device(rank % torch.cuda.device_count())

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Simulate Megatron parallel state initialization
    # In real Megatron: parallel_state.initialize_model_parallel(...)
    tp_size = 1
    pp_size = 1
    dp_size = world_size // (tp_size * pp_size)

    # Create model
    model = SimpleTransformer().cuda()

    # Wrap in DDP-like wrapper for Megatron
    # In real Megatron, you would use DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank % torch.cuda.device_count()],
        output_device=rank % torch.cuda.device_count(),
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    return model, optimizer, dp_size


def main():
    args = parse_args()

    # Get rank and world_size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {rank}] Initializing with world_size={world_size}")

    # Setup model and optimizer
    model, optimizer, dp_size = setup_megatron_simulator(rank, world_size, args)

    # Create dynamic manager
    # In real Megatron, you would pass the actual DDP wrapper
    dynamic_manager = DynamicDPManager(
        model=[model],  # Megatron uses list of models
        optimizer=optimizer,
        use_cpu_offload=True,
    )

    # Create simulator
    simulator = MegatronSimulator(world_size, rank, model, dynamic_manager)

    print(f"[Rank {rank}] Starting training loop...")

    # Training loop
    for iteration in range(args.sync_interval * 3):
        # Training step
        loss = simulator.train_step()

        if (iteration + 1) % 10 == 0:
            print(f"[Rank {rank}] Iteration {iteration + 1}, Loss: {loss:.4f}")

        # Periodic DP rebuild for elastic training
        if args.elastic and (iteration + 1) % args.sync_interval == 0:
            # In elastic mode, we would rebuild with potentially different DP size
            new_dp_size = min(args.target_dp_size, world_size)

            if rank == 0:
                print(f"\n{'=' * 50}")
                print("Elastic resize: Rebuilding DP groups")
                print(f"{'=' * 50}\n")

            simulator.rebuild_dp(new_dp_size)

            # Barrier to synchronize
            dist.barrier()

    print(f"[Rank {rank}] Training complete!")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Advanced Usage: Flexible DP for RL Training
# ============================================================================


class RLFlexibleDPTrainer:
    """
    RL training with dynamic allocation between train and rollout.

    This allows:
    - Training ranks: DP synchronized gradient updates
    - Rollout ranks: Offloaded for inference
    - Dynamic switching based on performance
    """

    def __init__(
        self,
        model,
        optimizer,
        train_ranks: list[int],
        rollout_ranks: list[int],
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_ranks = sorted(train_ranks)
        self.rollout_ranks = sorted(rollout_ranks)

        # Create subgroups
        self.train_group, _ = dist.new_subgroups_by_ranks(self.train_ranks)
        self.rollout_group, _ = dist.new_subgroups_by_ranks(self.rollout_ranks) if self.rollout_ranks else (None, None)

        self.current_mode = "train"
        self.rank = dist.get_rank()

    @property
    def is_train_rank(self) -> bool:
        return self.rank in self.train_ranks

    @property
    def is_rollout_rank(self) -> bool:
        return self.rank in self.rollout_ranks

    def switch_mode(self, mode: str):
        """
        Switch between 'train' and 'rollout' modes.

        Args:
            mode: 'train' or 'rollout'
        """
        self.current_mode = mode

        if mode == "rollout":
            # Rollout ranks: eval mode, no gradients
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Train ranks: training mode
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

    def train_step(self, data):
        """Training step with partial DP sync."""
        if not self.is_train_rank:
            return None

        # Forward
        outputs = self.model(**data)
        loss = outputs["loss"]

        # Backward
        loss.backward()

        # Sync gradients within train group only
        self.sync_gradients()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def sync_gradients(self):
        """Sync gradients only within train group."""
        if self.train_group is None:
            return

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.train_group)
                param.grad /= len(self.train_group.ranks)

    def generate_rollouts(self, prompts):
        """Generate rollouts (only on rollout ranks)."""
        if not self.is_rollout_rank:
            return None

        # Offload model to CPU during rollout if needed
        # ... inference code ...

        return []  # Placeholder

    def redistribute_resources(
        self,
        new_train_ranks: list[int],
        new_rollout_ranks: list[int],
        model_state: ModelStateSnapshot,
    ):
        """
        Redistribute resources between train and rollout.

        This is called when the resource allocation changes.
        """
        print(f"[Rank {self.rank}] Redistributing resources...")

        # Capture current state
        # ... capture code ...

        # Update rank lists
        self.train_ranks = sorted(new_train_ranks)
        self.rollout_ranks = sorted(new_rollout_ranks)

        # Recreate subgroups
        self.train_group, _ = dist.new_subgroups_by_ranks(self.train_ranks)
        if self.rollout_ranks:
            self.rollout_group, _ = dist.new_subgroups_by_ranks(self.rollout_ranks)

        # Broadcast model state to new groups
        # ... broadcast code ...

        print(f"[Rank {self.rank}] New configuration:")
        print(f"  Train ranks: {self.train_ranks}")
        print(f"  Rollout ranks: {self.rollout_ranks}")


def rl_example():
    """
    Example of RL training with flexible DP allocation.

    Assumes 4 GPUs (ranks 0-3):
    - Initially: all ranks train
    - Dynamic: 2 train, 2 rollout
    """
    rank = dist.get_rank()

    # Assume 4 GPUs
    # Scenario 1: All train (DP=4)
    train_ranks = [0, 1, 2, 3]

    trainer = RLFlexibleDPTrainer(
        model=None,  # Would be actual model
        optimizer=None,  # Would be actual optimizer
        train_ranks=train_ranks,
        rollout_ranks=[],
    )

    # Training phase: all ranks participate
    trainer.switch_mode("train")

    # When rollout is needed (e.g., every N iterations)
    if rank == 0:
        print("\nSwitching to mixed mode...")

    # Redistribute: ranks 0,1 train; ranks 2,3 rollout
    # Note: This would require actual dynamic DP rebuild in real implementation
    # trainer.redistribute_resources(
    #     new_train_ranks=[0, 1],
    #     new_rollout_ranks=[2, 3],
    #     model_state=None,
    # )


if __name__ == "__main__":
    main()
