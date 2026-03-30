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
Elastic engine extensions for ``verl/workers/engine``.

Sub-packages
------------
``fsdp``
    :class:`ElasticFSDPMixin` – dynamic DP-group rebuild for FSDP / FSDP2 engines.

``megatron``
    :class:`ElasticMegatronMixin` – dynamic DP-group rebuild for Megatron-Core engines.

Public API
----------
Import the factory function to compose elastic engines at runtime::

    from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls

    elastic_cls = get_elastic_engine_cls("fsdp", FSDPEngineWithLMHead)
    elastic_cls = get_elastic_engine_cls("megatron", MegatronEngineWithLMHead)

Architecture
============

This module follows the engine hierarchy in ``verl/workers/engine`` but adds
``rebuild_dp_group`` **without** modifying that package.  Per-strategy mixin
implementations live in dedicated sub-packages:

``verl.experimental.elastic_scheduling.engine.fsdp``
    ``ElasticFSDPMixin``  – patches ``device_mesh._dim_group_infos`` (FSDP/FSDP2)

``verl.experimental.elastic_scheduling.engine.megatron``
    ``ElasticMegatronMixin`` – patches Megatron ``parallel_state`` (mpu)

The factory :func:`get_elastic_engine_cls` dynamically creates a subclass that
combines the right mixin with any caller-supplied base engine class.

Usage
-----
::

    from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls

    # At worker initialisation, swap the engine class:
    elastic_cls = get_elastic_engine_cls(strategy, original_engine_cls)
    self.actor.engine = elastic_cls(...)

    # Later, during elastic scaling:
    self.actor.engine.rebuild_dp_group(new_world_ranks)
"""

from verl.experimental.elastic_scheduling.engine.fsdp import ElasticFSDPMixin, FSDP2DPRebuildManager
from verl.experimental.elastic_scheduling.engine.megatron import ElasticMegatronMixin, MegatronDPRebuildManager

# ---------------------------------------------------------------------------
# Strategy → Mixin mapping
# ---------------------------------------------------------------------------

_MIXIN_MAP: dict[str, type] = {
    "fsdp": ElasticFSDPMixin,
    "fsdp2": ElasticFSDPMixin,
    "megatron": ElasticMegatronMixin,
}

# Cache to avoid recreating the same combined class repeatedly.
_ENGINE_CLASS_CACHE: dict[tuple, type] = {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_elastic_engine_cls(strategy: str, base_cls: type) -> type:
    """
    Return a subclass of ``base_cls`` that additionally implements
    ``rebuild_dp_group`` for the given training strategy.

    The returned class is a dynamic combination of the appropriate elastic
    mixin and the caller-supplied base engine class::

        ElasticFSDPMixin    + FSDPEngineWithLMHead   →  ElasticFSDPEngineWithLMHead
        ElasticMegatronMixin + MegatronEngine        →  ElasticMegatronEngine

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


__all__ = [
    "ElasticFSDPMixin",
    "ElasticMegatronMixin",
    "FSDP2DPRebuildManager",
    "MegatronDPRebuildManager",
    "get_elastic_engine_cls",
]
