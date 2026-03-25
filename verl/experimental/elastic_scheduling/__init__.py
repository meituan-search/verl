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
Elastic Scheduling Module for VERL

This module provides elastic rollout and training scheduling capabilities:
- Dynamic resource allocation between rollout and training
- Congestion monitoring and auto-scaling
- Support for both FSDP2 and Megatron backends
"""

from .coordinator import (
    CongestionMetrics,
    CongestionMonitor,
    CoordinatorLoop,
    ResourceCoordinator,
)
from .elastic_rollouter import ElasticRollouterMixin
from .elastic_trainer import ElasticTrainerMixin
from .parameter_sync import ElasticCheckpointManager
from .resource_manager import (
    ElasticResourceConfig,
    ElasticResourceManager,
    HybridEngineMode,
    HybridEngineResource,
)

__all__ = [
    # Resource Management
    "ElasticResourceConfig",
    "ElasticResourceManager",
    "HybridEngineMode",
    "HybridEngineResource",
    # Coordination
    "CongestionMetrics",
    "CongestionMonitor",
    "ResourceCoordinator",
    "CoordinatorLoop",
    # Mixins (for composing with FullyAsyncRollouter/Trainer)
    "ElasticRollouterMixin",
    "ElasticTrainerMixin",
    # Sync
    "ElasticCheckpointManager",
]
