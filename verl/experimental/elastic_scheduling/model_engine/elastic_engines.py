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
Elastic engine extensions for dynamic Data Parallel group rebuilding.

Architecture
============

This module follows the engine hierarchy in ``verl/workers/engine`` but
adds ``rebuild_dp_group`` **without** modifying that package.  The approach:

1. Two mixins capture the per-strategy rebuild logic:

   ``ElasticFSDPMixin``     – patches ``device_mesh._dim_group_infos`` (FSDP/FSDP2)
   ``ElasticMegatronMixin`` – patches Megatron ``parallel_state`` (mpu)

2. ``get_elastic_engine_cls(strategy, base_cls)`` dynamically creates a
   subclass that combines the right mixin with the caller-supplied base
   engine class (e.g. ``FSDPEngineWithLMHead``).  This avoids listing every
   concrete base-class variant here.

Usage in ElasticActorWorker::

    from verl.experimental.elastic_scheduling.model_engine.elastic_engines import (
        get_elastic_engine_cls,
    )

    # At worker initialisation, swap the engine class used by EngineRegistry:
    elastic_cls = get_elastic_engine_cls(strategy, original_engine_cls)
    self.actor.engine = elastic_cls(...)

    # Later, during elastic scaling:
    self.actor.engine.rebuild_dp_group(new_world_ranks)
"""

import logging
import os

import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# FSDP mixin
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Megatron mixin
# ---------------------------------------------------------------------------


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
    Same as ``ElasticFSDPMixin``: all current ranks must call
    ``dist.new_group`` simultaneously.
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MIXIN_MAP: dict[str, type] = {
    "fsdp": ElasticFSDPMixin,
    "fsdp2": ElasticFSDPMixin,
    "megatron": ElasticMegatronMixin,
}

# Cache to avoid recreating the same combined class repeatedly.
_ENGINE_CLASS_CACHE: dict[tuple, type] = {}


def get_elastic_engine_cls(strategy: str, base_cls: type) -> type:
    """
    Return a subclass of ``base_cls`` that additionally implements
    ``rebuild_dp_group`` for the given training strategy.

    The returned class is a dynamic combination of the appropriate elastic
    mixin and the caller-supplied base engine class::

        ElasticFSDPMixin + FSDPEngineWithLMHead  →  new class
        ElasticMegatronMixin + MegatronEngine    →  new class

    Results are cached so the same combination always returns the same class
    object.

    Args:
        strategy: Training strategy string, e.g. ``"fsdp"``, ``"fsdp2"``,
            ``"megatron"``.
        base_cls: The concrete base engine class from ``verl.workers.engine``
            (e.g. ``FSDPEngine``, ``FSDPEngineWithLMHead``,
            ``MegatronEngineWithLMHead``).

    Returns:
        A new class ``Elastic<BaseName>`` that inherits from both the mixin
        and ``base_cls``, with the mixin listed first so its
        ``rebuild_dp_group`` takes precedence in the MRO.

    Raises:
        KeyError: If ``strategy`` has no registered elastic mixin.
    """
    if strategy not in _MIXIN_MAP:
        raise KeyError(f"No elastic engine mixin for strategy='{strategy}'. Supported: {sorted(_MIXIN_MAP)}")

    cache_key = (strategy, base_cls)
    if cache_key in _ENGINE_CLASS_CACHE:
        return _ENGINE_CLASS_CACHE[cache_key]

    mixin_cls = _MIXIN_MAP[strategy]
    elastic_cls = type(
        f"Elastic{base_cls.__name__}",
        (mixin_cls, base_cls),
        {},
    )
    _ENGINE_CLASS_CACHE[cache_key] = elastic_cls
    return elastic_cls
