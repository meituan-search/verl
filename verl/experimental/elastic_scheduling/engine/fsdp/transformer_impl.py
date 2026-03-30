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
Elastic FSDP/FSDP2 engine extension.

Provides:

``ElasticFSDPMixin``
    Adds ``rebuild_dp_group`` to any FSDP/FSDP2 engine without modifying
    ``verl/workers/engine``.

``FSDP2DPRebuildManager``
    Manages full DP-group rebuild for FSDP2: gather params to CPU, destroy
    old mesh, create new mesh, re-shard and broadcast parameters.

Usage
-----
Compose with a base engine class via the factory in the parent package::

    from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls
    elastic_cls = get_elastic_engine_cls("fsdp", FSDPEngineWithLMHead)
"""

import gc
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ElasticFSDPMixin:
    """
    Mixin that adds elastic DP-group rebuild to any FSDP / FSDP2 engine.

    Patching strategy
    ~~~~~~~~~~~~~~~~~
    FSDP routes all-reduce operations through the ``device_mesh`` (or
    ``ulysses_device_mesh`` when Ulysses SP is active).  The DP group is
    stored in ``mesh._dim_group_infos[0]``.  After calling
    ``dist.new_group(ranks=new_world_ranks)`` we overwrite that slot so that
    subsequent FSDP collectives use the new group.

    ``_dim_group_infos`` is a private PyTorch API – if it changes in a future
    release the fallback is safe: training continues with the old group until
    the next full re-initialisation.

    Collective-call contract
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``dist.new_group`` is a *collective*: **all** current ranks in the global
    process group must call it simultaneously.  Ranks absent from
    ``new_world_ranks`` participate in the collective and then return early.
    """

    def rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the FSDP data-parallel process group.

        Args:
            new_world_ranks: Ordered list of global ranks that form the new DP
                group.  All current ranks must call this method concurrently.
        """
        my_rank = dist.get_rank()
        logger.info(f"[ElasticFSDPEngine rank={my_rank}] rebuild_dp_group new_world_ranks={new_world_ranks}")

        # Collective: every rank in the global world participates.
        new_dp_group = dist.new_group(ranks=new_world_ranks)

        if my_rank not in new_world_ranks:
            logger.info(f"[ElasticFSDPEngine rank={my_rank}] removed from DP group")
            return

        # Patch the active mesh.  Prefer ulysses_device_mesh (SP active) over
        # the plain device_mesh.  DP is always dim 0 in both layouts:
        #   1-D FSDP mesh  ["fsdp"]         → dim 0 is the DP group
        #   2-D HSDP mesh  ["ddp", "fsdp"]  → dim 0 is the ddp (DP) group
        #   Ulysses mesh   ["dp", "sp"]     → dim 0 is the dp group
        mesh = getattr(self, "ulysses_device_mesh", None) or getattr(self, "device_mesh", None)
        if mesh is not None:
            try:
                mesh._dim_group_infos[0] = (new_dp_group, new_world_ranks)
                logger.info(
                    f"[ElasticFSDPEngine rank={my_rank}] device_mesh patched (new dp size={len(new_world_ranks)})"
                )
            except Exception as exc:
                logger.warning(
                    f"[ElasticFSDPEngine rank={my_rank}] Could not patch device_mesh: {exc}. "
                    "New group was created but mesh not updated."
                )
        else:
            logger.warning(f"[ElasticFSDPEngine rank={my_rank}] No device_mesh found; skip patch.")

        dist.barrier(group=new_dp_group)
        logger.info(f"[ElasticFSDPEngine rank={my_rank}] DP group rebuild complete (new size={len(new_world_ranks)})")


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
