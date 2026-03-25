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

Key Components:
    ElasticRollouter: Extends FullyAsyncRollouter with dynamic rollout DP
    ElasticTrainer: Extends FullyAsyncTrainer with dynamic training DP
    ElasticCoordinator: Monitors rates and triggers role switches (Ray actor)
    ElasticAgentLoopManager: AgentLoopManager with dynamic server management
    ElasticParameterSyncManager: Wraps CheckpointEngineManager for elastic replicas
    ElasticWorkerMixin: Mixin to add elastic switching to ActorRolloutRefWorker
    FSDP2DPRebuildManager: Handles FSDP2 DP group rebuilds
    MegatronDPRebuildManager: Handles Megatron DP group rebuilds

Legacy Components (for backward compatibility):
    CongestionMonitor, CongestionMetrics, ResourceCoordinator, CoordinatorLoop
    ElasticCheckpointManager, ElasticResourceConfig, ElasticResourceManager
"""

# New elastic scheduling components
from .coordinator import (
    # Legacy components (kept for backward compatibility)
    CongestionMetrics,
    CongestionMonitor,
    CoordinatorLoop,
    # New components
    ElasticCoordinator,
    ElasticResourceInfo,
    ResourceCoordinator,
    SuggestedAction,
)
from .elastic_agent_loop import (
    ElasticAgentLoopManager,
    ElasticGlobalRequestLoadBalancer,
)
from .elastic_param_sync import (
    ElasticParameterSyncManager,
    ElasticSyncStats,
)
from .elastic_rollouter import ElasticRollouter
from .elastic_trainer import ElasticTrainer
from .elastic_worker import (
    ElasticMode,
    ElasticWorkerMixin,
    ElasticWorkerState,
    FSDP2DPRebuildManager,
    MegatronDPRebuildManager,
)
from .parameter_sync import ElasticCheckpointManager
from .resource_manager import (
    ElasticResourceConfig,
    ElasticResourceManager,
    HybridEngineMode,
    HybridEngineResource,
)

__all__ = [
    # ---- New Core Components ----
    # Rollouter and Trainer
    "ElasticRollouter",
    "ElasticTrainer",
    # Coordinator (Ray actor)
    "ElasticCoordinator",
    "ElasticResourceInfo",
    # Agent Loop
    "ElasticAgentLoopManager",
    "ElasticGlobalRequestLoadBalancer",
    # Parameter Sync
    "ElasticParameterSyncManager",
    "ElasticSyncStats",
    # Worker Mixin and DP Rebuild
    "ElasticWorkerMixin",
    "ElasticWorkerState",
    "ElasticMode",
    "FSDP2DPRebuildManager",
    "MegatronDPRebuildManager",
    # ---- Resource Management ----
    "ElasticResourceConfig",
    "ElasticResourceManager",
    "HybridEngineMode",
    "HybridEngineResource",
    # ---- Legacy Components ----
    "CongestionMetrics",
    "CongestionMonitor",
    "ResourceCoordinator",
    "CoordinatorLoop",
    "SuggestedAction",
    "ElasticCheckpointManager",
]
