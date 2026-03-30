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
Elastic Scheduling for VERL

This module provides elastic scheduling capabilities for VERL training,
enabling dynamic resource allocation between training and rollout phases.

Main Components
---------------

Workers:
- ``HybridElasticActorWorker``: Worker supporting dynamic TRAIN ↔ ROLLOUT role
  switching within a collocated actor-rollout setup.

Engines:
- ``ElasticMegatronMixin``: Megatron engine mixin for DP group rebuilding.
- ``ElasticFSDPMixin``: FSDP/FSDP2 engine mixin for DP group rebuilding.

Managers:
- ``MegatronDPRebuildManager``: Manages Megatron DP group rebuilding.
- ``FSDP2DPRebuildManager``: Manages FSDP2 DP group rebuilding.

Quick Start
-----------

1. Using HybridElasticActorWorker (collocated actor + rollout)::

    from verl.experimental.elastic_scheduling import HybridElasticActorWorker

    worker = HybridElasticActorWorker(config, role="actor_rollout")
    worker.init_model()

    # Route this rank to training or rollout
    worker.switch_to_train(new_train_world_ranks=[0, 1, 2, 3], param_version=0)
    worker.switch_to_rollout(param_version=1)

2. Using Elastic Engine directly::

    from verl.experimental.elastic_scheduling.engine import get_elastic_engine_cls
    from verl.workers.engine.megatron import MegatronEngineWithLMHead

    ElasticEngineCls = get_elastic_engine_cls("megatron", MegatronEngineWithLMHead)
    engine = ElasticEngineCls(model_config, engine_config, optimizer_config, checkpoint_config)
    engine.initialize()
    engine.rebuild_dp_group(new_world_ranks)
"""

# Engines
from verl.experimental.elastic_scheduling.engine import (
    get_elastic_engine_cls,
)
from verl.experimental.elastic_scheduling.engine.fsdp import (
    ElasticFSDPMixin,
    FSDP2DPRebuildManager,
)
from verl.experimental.elastic_scheduling.engine.megatron import (
    ElasticMegatronMixin,
    MegatronDPRebuildManager,
)

# Workers
from verl.experimental.elastic_scheduling.hybrid_engine_workers import (
    ElasticMode,
    ElasticState,
    HybridElasticActorWorker,
)

__all__ = [
    # Workers
    "ElasticMode",
    "ElasticState",
    "HybridElasticActorWorker",
    # Engines
    "get_elastic_engine_cls",
    "ElasticMegatronMixin",
    "MegatronDPRebuildManager",
    "ElasticFSDPMixin",
    "FSDP2DPRebuildManager",
]
