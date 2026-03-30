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
      join or leave training.  The call is **delegated to the engine**, which
      must be one of the elastic engine classes from
      ``verl.experimental.elastic_scheduling.engine.elastic_engines``
      (e.g. ``ElasticFSDPEngine``, ``ElasticMegatronEngine``).  The worker
      itself has no strategy-specific knowledge.

  get_global_rank()
      Return this worker's global rank so that ElasticTrainer can compute
      the new DP world without holding a separate rank map.

All elastic methods are registered with Dispatch.ONE_TO_ALL so that
ElasticTrainer can broadcast the call to every worker in a worker group
with a single worker_group.execute_all() call.
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

    Design
    ------
    Strategy-specific DP rebuild logic is fully encapsulated inside the engine
    (see ``verl.experimental.elastic_scheduling.engine.fsdp`` /
    ``verl.experimental.elastic_scheduling.engine.megatron``).  This worker acts only as a thin dispatcher:

    * ``get_global_rank``   – returns ``dist.get_rank()`` for rank-list building.
    * ``rebuild_dp_group``  – delegates to ``self.actor.engine.rebuild_dp_group()``.

    The engine must be one of the ``Elastic*`` classes produced by
    ``get_elastic_engine_cls`` (from
    ``verl.experimental.elastic_scheduling.engine.elastic_engines``), which mixes
    in either ``ElasticFSDPMixin`` or ``ElasticMegatronMixin`` depending on the
    strategy.
    This means the worker never checks ``config.actor.strategy`` for the rebuild
    path.

    Both methods are decorated with ``Dispatch.ONE_TO_ALL`` so that
    ElasticTrainer can broadcast calls to every worker in a group via a single
    ``worker_group.execute_all()`` invocation.
    """

    # -------------------------------------------------------------------------
    # Rank introspection
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_global_rank(self) -> int:
        """
        Return this worker's global rank in the distributed process group.

        ElasticTrainer calls this to build the ``new_world_ranks`` list before
        triggering a DP group rebuild.

        Returns:
            int: ``torch.distributed`` global rank of this process.
        """
        return dist.get_rank()

    # -------------------------------------------------------------------------
    # DP group rebuild
    # -------------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rebuild_dp_group(self, new_world_ranks: list[int]) -> None:
        """
        Rebuild the data-parallel process group.

        Delegates to the engine object (``self.actor.engine``), which must
        implement ``rebuild_dp_group`` via one of the elastic engine mixins
        defined in ``verl.experimental.elastic_scheduling.engine.fsdp`` or
        ``verl.experimental.elastic_scheduling.engine.megatron``.

        Collective-call contract
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ``dist.new_group`` is a collective operation – **every** rank currently
        in the global process group must call ``rebuild_dp_group`` simultaneously.
        Ranks absent from ``new_world_ranks`` are expected to participate in the
        collective and then return early (handled inside the engine mixin).

        Args:
            new_world_ranks: Complete, ordered list of global ranks that form
                the new data-parallel group.
        """
        my_rank = dist.get_rank()
        logger.info(f"[ElasticActorWorker rank={my_rank}] rebuild_dp_group new_world_ranks={new_world_ranks}")

        if not callable(getattr(self.actor.engine, "rebuild_dp_group", None)):
            raise AttributeError(
                f"[ElasticActorWorker rank={my_rank}] engine {type(self.actor.engine).__name__} "
                "does not implement rebuild_dp_group().  Make sure the engine was "
                "created with get_elastic_engine_cls() from "
                "verl.experimental.elastic_scheduling.engine.elastic_engines."
            )

        self.actor.engine.rebuild_dp_group(new_world_ranks)
