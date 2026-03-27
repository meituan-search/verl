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
Elastic Actor Worker for VERL

Extends DetachActorWorker with all worker-side elastic operations:

  rebuild_dp_group(new_world_ranks)
      Rebuild the data-parallel communication group after elastic resources
      join or leave training.  Strategy-specific implementations for fsdp /
      fsdp2 and megatron are dispatched automatically.

  get_global_rank()
      Return this worker's global rank so that ElasticTrainer can compute
      the new DP world without holding a separate rank map.

All elastic methods are registered with Dispatch.ONE_TO_ALL so that
ElasticTrainer can broadcast the call to every worker in a worker group
with a single worker_group.execute_all() call, keeping the trainer free
of strategy-specific knowledge.
"""

import logging
import os

import torch.distributed as dist

from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ElasticActorWorker(DetachActorWorker):
    """
    Worker class that adds elastic DP management on top of DetachActorWorker.

    ElasticTrainer calls the three methods below (via worker_group.execute_all)
    whenever elastic resources switch between training and rollout mode:

      rebuild_dp_group   – rebuild the DP process group after membership change
      get_global_rank    – query this worker's global rank (for rank-list building)

    All methods are decorated with Dispatch.ONE_TO_ALL so that they are
    broadcast to every process in the worker group automatically.
    """

    # -------------------------------------------------------------------------
    # Rank introspection
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_global_rank(self) -> int:
        """
        Return this worker's global rank in the distributed process group.

        ElasticTrainer calls this to build the new_world_ranks list before
        triggering a DP group rebuild.

        Returns:
            int: torch.distributed global rank of this process.
        """
        return dist.get_rank()

    # -------------------------------------------------------------------------
    # DP group rebuild
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the data-parallel process group with a new set of ranks.

        Called by ElasticTrainer._trigger_rebuild_on_worker_group() after an
        elastic resource joins or leaves the training pool.  All workers that
        participate in the new DP group must call this method simultaneously.

        The implementation is delegated to a strategy-specific helper:
          - fsdp / fsdp2  →  _rebuild_dp_group_fsdp(new_world_ranks)
          - megatron       →  _rebuild_dp_group_megatron(new_world_ranks)

        Args:
            new_world_ranks: Complete, ordered list of global ranks that form
                the new data-parallel group.
        """
        strategy = self.config.actor.strategy
        logger.info(
            f"[ElasticActorWorker rank={dist.get_rank()}] rebuild_dp_group "
            f"strategy={strategy} new_world_ranks={new_world_ranks}"
        )

        if strategy in ("fsdp", "fsdp2"):
            self._rebuild_dp_group_fsdp(new_world_ranks)
        elif strategy == "megatron":
            self._rebuild_dp_group_megatron(new_world_ranks)
        else:
            raise NotImplementedError(
                f"[ElasticActorWorker] rebuild_dp_group: unsupported strategy '{strategy}'. "
                f"Supported: fsdp, fsdp2, megatron."
            )

    # -------------------------------------------------------------------------
    # Strategy-specific DP rebuild helpers (not exposed as remote methods)
    # -------------------------------------------------------------------------

    def _rebuild_dp_group_fsdp(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the DP process group for FSDP / FSDP2 strategies.

        Creates a new torch.distributed process group containing exactly the
        ranks in new_world_ranks and re-registers it on self.device_mesh so
        that subsequent FSDP all-reduce operations use the updated group.

        Args:
            new_world_ranks: Ordered list of global ranks in the new DP group.
        """
        my_rank = dist.get_rank()

        # All *current* ranks must call new_group together, even those that
        # are leaving the new group, to satisfy the NCCL collective contract.
        new_dp_group = dist.new_group(ranks=new_world_ranks)

        if my_rank not in new_world_ranks:
            # This worker is being removed from the DP group; nothing more to do.
            logger.info(f"[ElasticActorWorker rank={my_rank}] removed from DP group")
            return

        # Re-wire device_mesh so that the FSDP sharding group uses the new group.
        # self.device_mesh is a DeviceMesh; for a 1-D mesh the single group is
        # the DP group itself.  For 2-D (ddp × fsdp) the outer "ddp" dim is DP.
        if hasattr(self, "device_mesh") and self.device_mesh is not None:
            try:
                if self.device_mesh.ndim == 1:
                    # 1-D mesh: the single group IS the DP group
                    self.device_mesh._dim_group_infos[0] = (new_dp_group, new_world_ranks)
                elif self.device_mesh.ndim == 2:
                    # 2-D mesh (ddp × fsdp): update the outer "ddp" dim
                    self.device_mesh._dim_group_infos[0] = (new_dp_group, new_world_ranks)
                logger.info(
                    f"[ElasticActorWorker rank={my_rank}] FSDP device_mesh DP group updated "
                    f"(new size={len(new_world_ranks)})"
                )
            except Exception as e:
                logger.warning(
                    f"[ElasticActorWorker rank={my_rank}] Could not update device_mesh directly: {e}. "
                    f"New group created but mesh not patched."
                )

        # Barrier inside the new group to confirm all members are ready.
        dist.barrier(group=new_dp_group)
        logger.info(
            f"[ElasticActorWorker rank={my_rank}] FSDP DP group rebuild complete (group size={len(new_world_ranks)})"
        )

    def _rebuild_dp_group_megatron(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the DP process group for the Megatron strategy.

        Creates a new torch.distributed process group and updates Megatron's
        model-parallel utilities (mpu) so that data-parallel collectives use
        the rebuilt group.

        Args:
            new_world_ranks: Ordered list of global ranks in the new DP group.
        """
        try:
            from megatron.core import mpu
        except ImportError as exc:
            raise RuntimeError(
                "[ElasticActorWorker] Megatron-Core is not installed but "
                "strategy='megatron' was requested for rebuild_dp_group."
            ) from exc

        my_rank = dist.get_rank()

        # All current ranks must participate in new_group().
        new_dp_group = dist.new_group(ranks=new_world_ranks)

        if my_rank not in new_world_ranks:
            logger.info(f"[ElasticActorWorker rank={my_rank}] removed from Megatron DP group")
            return

        # Patch Megatron's internal DP group reference.
        try:
            mpu.set_data_parallel_group(new_dp_group)
            logger.info(
                f"[ElasticActorWorker rank={my_rank}] Megatron DP group updated (new size={len(new_world_ranks)})"
            )
        except AttributeError:
            # Older Megatron versions may not expose set_data_parallel_group;
            # fall back to patching the private attribute directly.
            if hasattr(mpu, "_DATA_PARALLEL_GROUP"):
                mpu._DATA_PARALLEL_GROUP = new_dp_group
                logger.info(f"[ElasticActorWorker rank={my_rank}] Patched mpu._DATA_PARALLEL_GROUP directly")
            else:
                logger.warning(
                    f"[ElasticActorWorker rank={my_rank}] Cannot patch Megatron DP group: "
                    f"mpu has no set_data_parallel_group or _DATA_PARALLEL_GROUP attribute."
                )

        dist.barrier(group=new_dp_group)
        logger.info(
            f"[ElasticActorWorker rank={my_rank}] Megatron DP group rebuild complete "
            f"(group size={len(new_world_ranks)})"
        )
