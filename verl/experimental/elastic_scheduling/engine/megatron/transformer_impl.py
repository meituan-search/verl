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
Elastic Megatron-Core engine extension.

Provides:

``ElasticMegatronMixin``
    Adds ``rebuild_dp_group`` to any Megatron-Core engine without modifying
    ``verl/workers/engine``.

``MegatronDPRebuildManager``
    Manages full DP-group rebuild for Megatron: capture state to CPU, destroy
    all parallel groups, reinitialize with new configuration, restore state.

Usage
-----
Compose with a base engine class via the factory in the parent package::

    from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls
    elastic_cls = get_elastic_engine_cls("megatron", MegatronEngineWithLMHead)
"""

import gc
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ElasticMegatronMixin:
    """
    Mixin that adds elastic DP-group rebuild to any Megatron-Core engine.

    Patching strategy
    ~~~~~~~~~~~~~~~~~
    Megatron-Core manages process groups through ``parallel_state`` (``mpu``).
    We update the DP group reference via ``set_data_parallel_group`` (recent
    versions) or by directly patching ``_DATA_PARALLEL_GROUP`` (older ones).

    Collective-call contract
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``dist.new_group`` is a *collective*: **all** current ranks in the global
    process group must call it simultaneously.  Ranks absent from
    ``new_world_ranks`` participate in the collective and then return early.
    """

    def rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the Megatron data-parallel process group.

        Args:
            new_world_ranks: Ordered list of global ranks that form the new DP
                group.  All current ranks must call this method concurrently.
        """
        try:
            from megatron.core import parallel_state as mpu
        except ImportError as exc:
            raise RuntimeError(
                "[ElasticMegatronEngine] Megatron-Core is not installed but "
                "strategy='megatron' was requested for rebuild_dp_group."
            ) from exc

        my_rank = dist.get_rank()
        logger.info(f"[ElasticMegatronEngine rank={my_rank}] rebuild_dp_group new_world_ranks={new_world_ranks}")

        # Collective: every rank in the global world participates.
        new_dp_group = dist.new_group(ranks=new_world_ranks)

        if my_rank not in new_world_ranks:
            logger.info(f"[ElasticMegatronEngine rank={my_rank}] removed from DP group")
            return

        # Patch mpu's internal DP group reference.
        if hasattr(mpu, "set_data_parallel_group"):
            mpu.set_data_parallel_group(new_dp_group)
            logger.info(
                f"[ElasticMegatronEngine rank={my_rank}] mpu DP group updated (new size={len(new_world_ranks)})"
            )
        elif hasattr(mpu, "_DATA_PARALLEL_GROUP"):
            mpu._DATA_PARALLEL_GROUP = new_dp_group
            logger.info(
                f"[ElasticMegatronEngine rank={my_rank}] Patched mpu._DATA_PARALLEL_GROUP directly "
                f"(new size={len(new_world_ranks)})"
            )
        else:
            logger.warning(
                f"[ElasticMegatronEngine rank={my_rank}] Cannot patch Megatron DP group: "
                "mpu exposes neither set_data_parallel_group nor _DATA_PARALLEL_GROUP."
            )

        dist.barrier(group=new_dp_group)
        logger.info(
            f"[ElasticMegatronEngine rank={my_rank}] DP group rebuild complete (new size={len(new_world_ranks)})"
        )


# ============================================================================
# Megatron DP Group Rebuild
# ============================================================================


class MegatronDPRebuildManager:
    """
    Manages dynamic rebuilding of Megatron-LM Data Parallel groups.

    Uses CPU memory as an intermediate buffer to avoid disk I/O.
    Works by:
    1. Capturing model + optimizer state to CPU
    2. Destroying all Megatron parallel groups
    3. Reinitializing with new configuration (including elastic workers)
    4. Restoring state from CPU memory

    Key difference from FSDP2: Megatron's parallel_state manages multiple
    process groups (TP, PP, DP, etc.) through a global singleton, so we
    must destroy and reinitialize the entire parallel state.
    """

    def __init__(self, model, optimizer=None):
        """
        Args:
            model: List of Megatron model chunks (list[LanguageModule])
            optimizer: MegatronOptimizer instance
        """
        self.model = model
        self.optimizer = optimizer

        # CPU state snapshots
        self._model_snapshot = None
        self._optimizer_snapshot = None
        self._model_on_gpu = True
        self._optimizer_on_gpu = True

    def capture_and_offload(self):
        """
        Capture model + optimizer state to CPU, then release GPU memory.
        """
        logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] Capturing state to CPU...")

        # Offload to CPU; state is referenced via self.model / self.optimizer
        self._offload_model_to_cpu()
        self._offload_optimizer_to_cpu()

        self._model_snapshot = True  # mark as captured
        self._optimizer_snapshot = True

        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] State captured and offloaded to CPU")

    def _offload_model_to_cpu(self):
        """Move model parameters to CPU."""
        if not self._model_on_gpu:
            return

        unwrapped = self.model[0].module if hasattr(self.model[0], "module") else self.model[0]
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        for param in unwrapped.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad = param.grad.cpu()

        self._model_on_gpu = False

    def _offload_optimizer_to_cpu(self):
        """Move optimizer state to CPU."""
        if not self._optimizer_on_gpu or self.optimizer is None:
            return

        if hasattr(self.optimizer, "offload_to_cpu"):
            self.optimizer.offload_to_cpu()
        else:
            # Manual offload for standard optimizers
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        self._optimizer_on_gpu = False

    def destroy_parallel_groups(self):
        """Destroy all Megatron parallel groups."""
        try:
            from megatron.core import parallel_state

            logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] Destroying parallel groups...")
            parallel_state.destroy_model_parallel()
        except ImportError:
            logger.warning("[MegatronRebuild] Megatron-LM not available")

    def reinitialize_parallel_groups(
        self,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        new_dp_size: int,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
    ):
        """
        Reinitialize Megatron parallel state with new DP configuration.

        Args:
            tensor_model_parallel_size: TP size (unchanged)
            pipeline_model_parallel_size: PP size (unchanged)
            new_dp_size: New DP size (after adding/removing elastic workers)
            context_parallel_size: CP size (unchanged)
            expert_model_parallel_size: EP size (unchanged)
        """
        try:
            from megatron.core import parallel_state

            logger.info(
                f"[MegatronRebuild][Rank {dist.get_rank()}] "
                f"Reinitializing: tp={tensor_model_parallel_size}, "
                f"pp={pipeline_model_parallel_size}, dp={new_dp_size}"
            )

            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
            )
        except ImportError:
            logger.warning("[MegatronRebuild] Megatron-LM not available")

    def restore_from_cpu(self):
        """Restore model and optimizer state from CPU memory to GPU."""
        logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] Restoring state from CPU to GPU...")

        # Load model back to GPU
        if not self._model_on_gpu and self._model_snapshot is not None:
            unwrapped = self.model[0].module if hasattr(self.model[0], "module") else self.model[0]
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            for param in unwrapped.parameters():
                param.data = param.data.cuda()

            self._model_on_gpu = True

        # Load optimizer back to GPU
        if not self._optimizer_on_gpu and self.optimizer is not None:
            if hasattr(self.optimizer, "load_from_cpu"):
                self.optimizer.load_from_cpu()
            else:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self._optimizer_on_gpu = True

        torch.cuda.empty_cache()
        logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] State restored to GPU")

    def sync_new_member_params(self, new_member_ranks: list[int]):
        """
        Broadcast parameters from existing ranks to newly joined elastic ranks.

        This is called after reinitializing parallel groups to ensure new members
        have the correct parameter values.

        Args:
            new_member_ranks: List of ranks that are new to the DP group
        """
        try:
            from megatron.core import parallel_state

            dp_group = parallel_state.get_data_parallel_group()
            dp_rank = parallel_state.get_data_parallel_rank()

            logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] Syncing params to new members, dp_rank={dp_rank}")

            # Broadcast from dp_rank 0 to all DP members
            # (existing dp rank 0 has the latest params from CPU snapshot)
            unwrapped = self.model[0].module if hasattr(self.model[0], "module") else self.model[0]
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            for param in unwrapped.parameters():
                dist.broadcast(param.data, src=0, group=dp_group)

        except ImportError:
            logger.warning("[MegatronRebuild] Megatron-LM not available")

    def rebuild(
        self,
        new_world_size: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        new_member_ranks: Optional[list[int]] = None,
    ):
        """
        Full Megatron DP group rebuild sequence.

        Args:
            new_world_size: New total world size after adding elastic workers
            tensor_model_parallel_size: TP size
            pipeline_model_parallel_size: PP size
            new_member_ranks: Global ranks of newly joined elastic workers
        """
        new_dp_size = new_world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

        # Step 1: Capture and offload
        self.capture_and_offload()

        # Step 2: Destroy old groups
        self.destroy_parallel_groups()

        # Step 3: Barrier (all ranks including elastic workers must reach here)
        if dist.is_initialized():
            dist.barrier()

        # Step 4: Reinitialize parallel groups
        self.reinitialize_parallel_groups(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            new_dp_size=new_dp_size,
        )

        # Step 5: Barrier after reinitialization
        if dist.is_initialized():
            dist.barrier()

        # Step 6: Restore from CPU
        self.restore_from_cpu()

        # Step 7: Sync new members
        if new_member_ranks:
            self.sync_new_member_params(new_member_ranks)

        # Step 8: Final barrier
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] Full rebuild complete, new dp_size={new_dp_size}")
