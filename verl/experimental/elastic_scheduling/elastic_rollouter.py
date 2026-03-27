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
Elastic Rollouter for VERL

Thin extension of FullyAsyncRollouter that:
- Overrides _init_async_rollout_manager to use ElasticAgentLoopManager
- Delegates all elastic replica lifecycle to ElasticAgentLoopManager
- Keeps ElasticRollouter focused on sample production, not server management
"""

import logging
from typing import TYPE_CHECKING

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType

if TYPE_CHECKING:
    from verl.experimental.elastic_scheduling.elastic_agent_loop import ElasticAgentLoopManager
    from verl.workers.rollout.replica import RolloutReplica

logger = logging.getLogger(__name__)


class ElasticRollouter(FullyAsyncRollouter):
    """
    Elastic Rollouter – thin wrapper over FullyAsyncRollouter.

    The only responsibility added here is swapping in ElasticAgentLoopManager
    (which understands how to dynamically add/remove rollout servers) in place
    of the standard FullyAsyncAgentLoopManager.

    All elastic replica tracking, versioning, and statistics live in
    ElasticAgentLoopManager.  ElasticRollouter merely forwards the calls.

    Architecture:
        ElasticRollouter  (Ray actor – sample production loop)
            └── ElasticAgentLoopManager  (server pool + LB management)
                    ├── Fixed Rollout Replicas  (always active)
                    └── Elastic Rollout Replicas  (dynamically added/removed)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=device_name,
        )
        logger.info("[ElasticRollouter] Initialised (elastic server management via ElasticAgentLoopManager)")

    # -------------------------------------------------------------------------
    # Override: swap in ElasticAgentLoopManager
    # -------------------------------------------------------------------------

    async def _init_async_rollout_manager(self):
        """Use ElasticAgentLoopManager instead of FullyAsyncAgentLoopManager."""
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        assert self.config.actor_rollout_ref.rollout.mode == "async"

        self.async_rollout_mode = True
        self.async_rollout_manager: ElasticAgentLoopManager = await ElasticAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        logger.info(
            f"[ElasticRollouter] ElasticAgentLoopManager initialised with "
            f"{len(self.async_rollout_manager.rollout_replicas)} fixed replicas"
        )

    # -------------------------------------------------------------------------
    # Elastic replica management – thin delegation to async_rollout_manager
    # -------------------------------------------------------------------------

    async def add_elastic_replica(
        self,
        resource_id: str,
        replica: "RolloutReplica",
        param_version: int,
    ) -> bool:
        """
        Add an elastic rollout replica.  Fully handled by ElasticAgentLoopManager.

        Also adjusts max_concurrent_samples to account for the new replica.
        """
        ok = await self.async_rollout_manager.add_elastic_replica(resource_id, replica, param_version)
        if ok and self.max_concurrent_samples is not None:
            async with self.lock:
                self.max_concurrent_samples += 16  # ~16 slots per replica
        return ok

    async def remove_elastic_replica(self, resource_id: str) -> bool:
        """
        Remove an elastic rollout replica.  Fully handled by ElasticAgentLoopManager.

        Also adjusts max_concurrent_samples.
        """
        ok = await self.async_rollout_manager.remove_elastic_replica(resource_id)
        if ok and self.max_concurrent_samples is not None:
            async with self.lock:
                self.max_concurrent_samples = max(16, self.max_concurrent_samples - 16)
        return ok

    def update_elastic_replica_version(self, resource_id: str, param_version: int) -> None:
        """Update the param version for an elastic replica after a sync."""
        self.async_rollout_manager.update_elastic_replica_version(resource_id, param_version)

    # -------------------------------------------------------------------------
    # Statistics / introspection – delegate to async_rollout_manager
    # -------------------------------------------------------------------------

    async def get_elastic_statistics(self) -> dict:
        """Combined rollout + elastic statistics."""
        base_stats = await self.get_statistics()
        elastic_stats = self.async_rollout_manager.get_elastic_statistics()
        return {**base_stats, **elastic_stats}

    def get_num_active_replicas(self) -> int:
        """Total active rollout replicas (fixed + elastic)."""
        return self.async_rollout_manager.get_active_server_count()

    def get_elastic_replicas_info(self) -> list[dict]:
        """Metadata for all elastic replicas."""
        return self.async_rollout_manager.get_elastic_replicas_info()

    def get_total_produced_samples(self) -> int:
        """Total samples produced (uses base class counter)."""
        return self.total_generated_samples
