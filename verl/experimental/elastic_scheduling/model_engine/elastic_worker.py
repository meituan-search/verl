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
Elastic Worker for VERL

ElasticActorRolloutRefWorker extends ActorRolloutRefWorker to support dynamic
switching between rollout and training modes. This is the core building block
for elastic scheduling.

Key capabilities:
1. Dual-mode initialization: both actor engine and rollout engine are initialized
2. Mode switching via sleep/wake_up mechanism
3. DP group rebuild support for FSDP2 and Megatron after mode switch
4. Parameter synchronization hooks for elastic resource joining
"""

import asyncio
import gc
import logging
import time
from enum import Enum, auto
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ElasticMode(Enum):
    """Operation mode of an elastic worker."""

    ROLLOUT = auto()  # Worker is acting as a rollout server
    TRAIN = auto()  # Worker is participating in training
    SWITCHING = auto()  # Worker is in the middle of a mode switch


class ElasticWorkerState:
    """State tracking for an elastic worker."""

    def __init__(self, resource_id: str, initial_mode: ElasticMode = ElasticMode.ROLLOUT):
        self.resource_id = resource_id
        self.current_mode = initial_mode
        self.param_version: int = -1  # -1 means not yet synced
        self.last_switch_time: float = 0.0
        self.total_switches: int = 0
        self.is_healthy: bool = True

    def record_switch(self, new_mode: ElasticMode):
        self.current_mode = new_mode
        self.last_switch_time = time.time()
        self.total_switches += 1

    def update_param_version(self, version: int):
        self.param_version = version


# ============================================================================
# FSDP2 DP Group Rebuild
# ============================================================================


class FSDP2DPRebuildManager:
    """
    Manages dynamic rebuilding of FSDP2 Data Parallel groups.

    When elastic resources join/leave the training pool, the DP group
    must be rebuilt to reflect the new world of training workers.

    Design:
    - Gather full (unsharded) parameters before destroying old DP group
    - Destroy old device_mesh and associated process groups
    - Create new device_mesh with updated dp_size
    - Re-shard model parameters according to new mesh
    - Broadcast parameters from rank 0 of new DP group to all ranks

    Constraints:
    - TP and PP dimensions remain unchanged during elastic scaling
    - Only the DP dimension changes
    - All participating workers must synchronize at each step
    """

    def __init__(self, model, optimizer=None, tp_size: int = 1, pp_size: int = 1):
        """
        Args:
            model: The FSDP2-wrapped model
            optimizer: The optimizer (if any)
            tp_size: Tensor parallel size (fixed, not changed by elastic scaling)
            pp_size: Pipeline parallel size (fixed, not changed by elastic scaling)
        """
        self.model = model
        self.optimizer = optimizer
        self.tp_size = tp_size
        self.pp_size = pp_size

        # CPU state snapshots
        self._cpu_params: Optional[dict] = None
        self._cpu_optim_state: Optional[dict] = None
        self._params_gathered: bool = False

    def gather_full_params_to_cpu(self):
        """
        Gather fully unsharded model parameters to CPU memory.

        For FSDP2, parameters are sharded across the DP group. We need to
        unshard them before rebuilding the DP group.
        """
        logger.info(f"[FSDP2Rebuild][Rank {dist.get_rank()}] Gathering full params to CPU...")

        self._cpu_params = {}

        # Use FSDP2's unshard context to get full params
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel

            # Gather full state dict
            with FullyShardedDataParallel.state_dict_type(
                self.model,
                state_dict_type=FullyShardedDataParallel.StateDictType.FULL_STATE_DICT,
            ):
                full_sd = self.model.state_dict()

            # Move to CPU
            for name, param in full_sd.items():
                self._cpu_params[name] = param.detach().cpu()

        except Exception:
            # Fallback: try FSDP2 style (torch >= 2.4)
            try:
                # Summon full params
                with self.model.summon_full_params(self.model, writeback=False, offload_to_cpu=True):
                    for name, param in self.model.named_parameters():
                        self._cpu_params[name] = param.detach().cpu().clone()
            except Exception as e:
                logger.warning(f"[FSDP2Rebuild] Could not gather full params via FSDP2 API: {e}")
                # Last resort: direct parameter copy (for rank 0 only in broadcast scenario)
                for name, param in self.model.named_parameters():
                    self._cpu_params[name] = param.detach().cpu().clone()

        self._params_gathered = True
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[FSDP2Rebuild][Rank {dist.get_rank()}] Gathered {len(self._cpu_params)} params to CPU")

    def capture_optimizer_state(self):
        """Save optimizer state to CPU memory."""
        if self.optimizer is None:
            return

        logger.info(f"[FSDP2Rebuild][Rank {dist.get_rank()}] Capturing optimizer state to CPU...")
        self._cpu_optim_state = {}

        try:
            optim_state = self.optimizer.state_dict()
            # Deep copy state to CPU
            for key, value in optim_state["state"].items():
                self._cpu_optim_state[key] = {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in value.items()
                }
        except Exception as e:
            logger.warning(f"[FSDP2Rebuild] Failed to capture optimizer state: {e}")

    def destroy_dp_groups(self):
        """
        Destroy existing DP-related process groups.

        This does NOT destroy TP/PP groups, only the DP dimension.
        """
        logger.info(f"[FSDP2Rebuild][Rank {dist.get_rank()}] Destroying DP groups...")
        # Note: PyTorch doesn't have a direct API to destroy individual sub-process-groups
        # The device_mesh's process groups will be garbage collected after we create new ones
        # We rely on dist.barrier() to ensure all ranks reach the same point before proceeding

    def rebuild(self, new_world_ranks: list[int], global_rank: int):
        """
        Full DP group rebuild sequence.

        Args:
            new_world_ranks: List of global ranks participating in the new DP group
            global_rank: Current rank's global rank in dist world
        """
        from torch.distributed.device_mesh import init_device_mesh

        new_dp_size = len(new_world_ranks) // (self.tp_size * self.pp_size)

        logger.info(
            f"[FSDP2Rebuild][Rank {global_rank}] Rebuilding: "
            f"new_dp_size={new_dp_size}, tp={self.tp_size}, pp={self.pp_size}"
        )

        # Step 1: Gather params if not already done
        if not self._params_gathered:
            self.gather_full_params_to_cpu()

        # Step 2: Capture optimizer state
        self.capture_optimizer_state()

        # Step 3: Barrier - all ranks must be here
        if dist.is_initialized():
            dist.barrier()

        # Step 4: Create new sub-process-group for new DP members
        # We create a new process group containing only the new_world_ranks
        new_group = dist.new_group(ranks=new_world_ranks)

        # Step 5: Create new device_mesh
        new_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(new_dp_size, self.tp_size),
            mesh_dim_names=["dp", "tp"],
            device_ids=new_world_ranks,
        )

        # Step 6: Restore model with new mesh
        # This requires re-applying FSDP2 or redistributing shards
        self._restore_model_with_new_mesh(new_mesh, new_group, global_rank in new_world_ranks)

        # Step 7: Barrier after rebuild
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"[FSDP2Rebuild][Rank {global_rank}] DP rebuild complete")

        return new_mesh

    def _restore_model_with_new_mesh(self, new_mesh, new_group, is_participating: bool):
        """
        Restore model parameters with new device mesh.

        For newly joining ranks: receive parameters from existing ranks via broadcast.
        For existing ranks: re-shard parameters according to new DP size.
        """
        if not is_participating:
            return

        # Broadcast full params from rank 0 of new group to all group members
        if self._cpu_params is not None:
            for name, cpu_param in self._cpu_params.items():
                param_tensor = cpu_param.cuda()
                dist.broadcast(param_tensor, src=0, group=new_group)

                # Find the corresponding model parameter and update
                for model_name, model_param in self.model.named_parameters():
                    if model_name == name:
                        with torch.no_grad():
                            if model_param.shape == param_tensor.shape:
                                model_param.data.copy_(param_tensor)
                            else:
                                # Parameter is sharded, need to extract our shard
                                dp_rank = dist.get_rank(group=new_group)
                                new_dp_size = dist.get_world_size(group=new_group) // self.tp_size
                                shard_size = param_tensor.numel() // new_dp_size
                                shard_start = dp_rank * shard_size
                                shard_end = shard_start + shard_size
                                model_param.data.copy_(
                                    param_tensor.view(-1)[shard_start:shard_end].view(model_param.shape)
                                )
                        break

        # Restore optimizer state if available
        if self._cpu_optim_state is not None and self.optimizer is not None:
            try:
                state_dict = self.optimizer.state_dict()
                for key, value in self._cpu_optim_state.items():
                    if key in state_dict["state"]:
                        for k, v in value.items():
                            if isinstance(v, torch.Tensor):
                                state_dict["state"][key][k] = v.cuda()
                            else:
                                state_dict["state"][key][k] = v
                self.optimizer.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(f"[FSDP2Rebuild] Failed to restore optimizer state: {e}")

    def clear_cpu_buffers(self):
        """Release CPU memory buffers."""
        self._cpu_params = None
        self._cpu_optim_state = None
        self._params_gathered = False
        gc.collect()


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

        # CPU state snapshots (reuse from dynamic_dp_manager)
        self._model_snapshot = None
        self._optimizer_snapshot = None
        self._model_on_gpu = True
        self._optimizer_on_gpu = True

    def capture_and_offload(self):
        """
        Capture model + optimizer state to CPU, then release GPU memory.
        """
        from verl.experimental.elastic_scheduling.model_engine.dynamic_dp_manager import (
            ModelStateSnapshot,
            OptimizerStateSnapshot,
        )

        logger.info(f"[MegatronRebuild][Rank {dist.get_rank()}] Capturing state to CPU...")

        self._model_snapshot = ModelStateSnapshot.from_model(self.model)
        self._optimizer_snapshot = OptimizerStateSnapshot.from_optimizer(self.optimizer)

        # Offload to CPU
        self._offload_model_to_cpu()
        self._offload_optimizer_to_cpu()

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

            self._model_snapshot.to_model(self.model, device="cuda")
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


# ============================================================================
# Elastic Worker Mixin
# ============================================================================


class ElasticWorkerMixin:
    """
    Mixin that adds elastic scheduling capabilities to ActorRolloutRefWorker.

    This mixin provides:
    1. Mode switching (Rollout <-> Train)
    2. DP group rebuild for FSDP2 and Megatron
    3. Parameter synchronization hooks

    Usage:
        class MyElasticWorker(ElasticWorkerMixin, ActorRolloutRefWorker):
            pass

    The mixin assumes the base class has:
    - self.actor: training engine
    - self.rollout: rollout engine
    - self.config: configuration
    """

    def init_elastic_state(self, resource_id: str, initial_mode: ElasticMode = ElasticMode.ROLLOUT):
        """
        Initialize elastic state. Must be called after __init__.

        Args:
            resource_id: Unique identifier for this elastic resource
            initial_mode: Initial mode (default: ROLLOUT)
        """
        self.elastic_state = ElasticWorkerState(resource_id, initial_mode)
        self._fsdp2_rebuild_manager: Optional[FSDP2DPRebuildManager] = None
        self._megatron_rebuild_manager: Optional[MegatronDPRebuildManager] = None
        self._mode_switch_lock = asyncio.Lock()

        # Initialize DP rebuild manager based on strategy
        strategy = getattr(self.config, "actor", {})
        if hasattr(strategy, "strategy"):
            actor_strategy = strategy.strategy
        else:
            actor_strategy = getattr(strategy, "get", lambda k, d: d)("strategy", "fsdp2")

        self._actor_strategy = actor_strategy
        logger.info(f"[ElasticWorker] {resource_id} initialized in {initial_mode.name} mode, strategy={actor_strategy}")

    async def switch_to_train(
        self,
        new_train_world_ranks: list[int],
        param_version: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> bool:
        """
        Switch this elastic worker from Rollout mode to Train mode.

        Steps:
        1. Stop accepting new rollout requests
        2. Wait for in-flight rollout requests to complete
        3. Sleep rollout engine (release KV cache and inference weights)
        4. Rebuild DP communication group with new training workers
        5. Update param version tracking

        Args:
            new_train_world_ranks: Global ranks of all workers in the new train DP group
            param_version: Current parameter version from trainer
            tensor_parallel_size: TP size (fixed)
            pipeline_parallel_size: PP size (fixed)

        Returns:
            True if switch succeeded, False otherwise
        """
        async with self._mode_switch_lock:
            if self.elastic_state.current_mode == ElasticMode.TRAIN:
                logger.warning(f"[ElasticWorker] {self.elastic_state.resource_id} already in TRAIN mode")
                return True

            logger.info(
                f"[ElasticWorker] {self.elastic_state.resource_id}: "
                f"Switching ROLLOUT -> TRAIN, param_version={param_version}"
            )
            self.elastic_state.current_mode = ElasticMode.SWITCHING

            try:
                # Step 1: Sleep rollout engine to free GPU memory
                await self._sleep_rollout_for_training()

                # Step 2: Rebuild DP group for training
                await self._rebuild_dp_for_training(
                    new_train_world_ranks=new_train_world_ranks,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                )

                # Step 3: Update state
                self.elastic_state.record_switch(ElasticMode.TRAIN)
                self.elastic_state.update_param_version(param_version)

                logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Successfully switched to TRAIN mode")
                return True

            except Exception as e:
                logger.error(f"[ElasticWorker] {self.elastic_state.resource_id}: Failed to switch to TRAIN: {e}")
                self.elastic_state.current_mode = ElasticMode.ROLLOUT  # Rollback
                self.elastic_state.is_healthy = False
                return False

    async def switch_to_rollout(
        self,
        new_rollout_world_ranks: list[int],
        param_version: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> bool:
        """
        Switch this elastic worker from Train mode to Rollout mode.

        Steps:
        1. Finish current training step (if any)
        2. Save model state (optimizer state stays in actor engine)
        3. Wake up rollout engine
        4. Sync latest parameters to rollout engine
        5. Update rollout DP group

        Args:
            new_rollout_world_ranks: Global ranks for the rollout inference group
            param_version: Current parameter version to sync
            tensor_parallel_size: TP size for rollout
            pipeline_parallel_size: PP size for rollout

        Returns:
            True if switch succeeded, False otherwise
        """
        async with self._mode_switch_lock:
            if self.elastic_state.current_mode == ElasticMode.ROLLOUT:
                logger.warning(f"[ElasticWorker] {self.elastic_state.resource_id} already in ROLLOUT mode")
                return True

            logger.info(
                f"[ElasticWorker] {self.elastic_state.resource_id}: "
                f"Switching TRAIN -> ROLLOUT, param_version={param_version}"
            )
            self.elastic_state.current_mode = ElasticMode.SWITCHING

            try:
                # Step 1: Offload actor engine to CPU (keep params in memory)
                await self._offload_actor_for_rollout()

                # Step 2: Wake up rollout engine
                await self._wake_up_rollout_for_inference(new_rollout_world_ranks)

                # Step 3: Update state (params will be synced by ElasticParamSync)
                self.elastic_state.record_switch(ElasticMode.ROLLOUT)
                self.elastic_state.update_param_version(param_version)

                logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Successfully switched to ROLLOUT mode")
                return True

            except Exception as e:
                logger.error(f"[ElasticWorker] {self.elastic_state.resource_id}: Failed to switch to ROLLOUT: {e}")
                self.elastic_state.current_mode = ElasticMode.TRAIN  # Rollback
                self.elastic_state.is_healthy = False
                return False

    async def _sleep_rollout_for_training(self):
        """
        Sleep the rollout engine to free GPU memory for training.

        Uses the existing sleep/wake_up mechanism from checkpoint_engine.
        """
        if not hasattr(self, "rollout") or self.rollout is None:
            logger.debug("[ElasticWorker] No rollout engine to sleep")
            return

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Sleeping rollout engine...")

        # Level 2: fully release weights and KV cache
        if hasattr(self.rollout, "sleep"):
            await self.rollout.sleep(level=2)
        elif hasattr(self.rollout, "_sleep_hybrid"):
            await self.rollout._sleep_hybrid()
        else:
            logger.warning("[ElasticWorker] Rollout engine has no sleep() method")

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Rollout engine sleeping")

    async def _wake_up_rollout_for_inference(self, world_ranks: list[int]):
        """
        Wake up the rollout engine for inference.

        After waking up, the rollout engine will accept new requests.
        Parameters need to be synced separately via ElasticParamSync.
        """
        if not hasattr(self, "rollout") or self.rollout is None:
            logger.debug("[ElasticWorker] No rollout engine to wake up")
            return

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Waking up rollout engine...")

        if hasattr(self.rollout, "wake_up"):
            await self.rollout.wake_up()
        elif hasattr(self.rollout, "_wake_up_hybrid"):
            await self.rollout._wake_up_hybrid()
        else:
            logger.warning("[ElasticWorker] Rollout engine has no wake_up() method")

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Rollout engine awake")

    async def _offload_actor_for_rollout(self):
        """
        Offload actor engine to CPU to free GPU memory for rollout.

        The actor's parameters stay in CPU memory (not disk) so they can be
        quickly restored when switching back to training mode.
        """
        if not hasattr(self, "actor") or self.actor is None:
            logger.debug("[ElasticWorker] No actor engine to offload")
            return

        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Offloading actor to CPU...")

        try:
            # Try FSDP2 offload
            from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu, offload_fsdp_optimizer

            if hasattr(self.actor, "engine") and hasattr(self.actor.engine, "module"):
                offload_fsdp_model_to_cpu(self.actor.engine.module)
            elif hasattr(self.actor, "actor_module"):
                offload_fsdp_model_to_cpu(self.actor.actor_module)

            if hasattr(self.actor, "optimizer"):
                offload_fsdp_optimizer(self.actor.optimizer)

        except ImportError:
            # Fallback: manual CPU offload
            if hasattr(self.actor, "engine"):
                for param in self.actor.engine.parameters():
                    param.data = param.data.cpu()
                    if param.grad is not None:
                        param.grad = param.grad.cpu()

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[ElasticWorker] {self.elastic_state.resource_id}: Actor offloaded to CPU")

    async def _rebuild_dp_for_training(
        self,
        new_train_world_ranks: list[int],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """
        Rebuild DP communication group for training.

        Routes to FSDP2 or Megatron rebuild based on actor strategy.
        """
        strategy = self._actor_strategy

        if strategy in ("fsdp2", "fsdp"):
            await self._rebuild_fsdp2_dp(new_train_world_ranks, tensor_parallel_size, pipeline_parallel_size)
        elif strategy == "megatron":
            await self._rebuild_megatron_dp(new_train_world_ranks, tensor_parallel_size, pipeline_parallel_size)
        else:
            logger.warning(f"[ElasticWorker] Unknown actor strategy '{strategy}', skipping DP rebuild")

    async def _rebuild_fsdp2_dp(
        self,
        new_world_ranks: list[int],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """Rebuild FSDP2 DP group."""
        model = getattr(getattr(self, "actor", None), "engine", None)
        if model is None:
            model = getattr(getattr(self, "actor", None), "actor_module", None)

        optimizer = getattr(getattr(self, "actor", None), "optimizer", None)

        if model is None:
            logger.warning("[ElasticWorker] Could not find actor model for FSDP2 rebuild")
            return

        if self._fsdp2_rebuild_manager is None:
            self._fsdp2_rebuild_manager = FSDP2DPRebuildManager(
                model=model,
                optimizer=optimizer,
                tp_size=tensor_parallel_size,
                pp_size=pipeline_parallel_size,
            )

        global_rank = dist.get_rank()
        self._fsdp2_rebuild_manager.rebuild(new_world_ranks, global_rank)

    async def _rebuild_megatron_dp(
        self,
        new_world_ranks: list[int],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        """Rebuild Megatron DP group."""
        model = getattr(getattr(self, "actor", None), "actor_module", None)
        optimizer = getattr(getattr(self, "actor", None), "optimizer", None)

        if model is None:
            logger.warning("[ElasticWorker] Could not find actor model for Megatron rebuild")
            return

        if self._megatron_rebuild_manager is None:
            self._megatron_rebuild_manager = MegatronDPRebuildManager(
                model=model,
                optimizer=optimizer,
            )

        # Identify newly joining ranks
        try:
            from megatron.core import parallel_state

            current_dp_group = parallel_state.get_data_parallel_group()
            current_dp_ranks = list(dist.get_process_group_ranks(current_dp_group))
        except Exception:
            current_dp_ranks = []

        new_member_ranks = [r for r in new_world_ranks if r not in current_dp_ranks]

        new_world_size = len(new_world_ranks)
        self._megatron_rebuild_manager.rebuild(
            new_world_size=new_world_size,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            new_member_ranks=new_member_ranks if new_member_ranks else None,
        )

    def get_elastic_state(self) -> dict:
        """Get current elastic state as a dictionary."""
        return {
            "resource_id": self.elastic_state.resource_id,
            "current_mode": self.elastic_state.current_mode.name,
            "param_version": self.elastic_state.param_version,
            "last_switch_time": self.elastic_state.last_switch_time,
            "total_switches": self.elastic_state.total_switches,
            "is_healthy": self.elastic_state.is_healthy,
        }
