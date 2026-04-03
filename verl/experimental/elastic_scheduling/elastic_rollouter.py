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
- Holds a reference to the elastic worker group so hybrid replicas can be
  created at initialisation time
- Delegates all elastic replica lifecycle to ElasticAgentLoopManager
- Keeps ElasticRollouter focused on sample production, not server management

Resource model
--------------
ElasticRollouter manages two kinds of rollout resources simultaneously:

  Fixed standalone replicas  (managed by base class)
    Always-on rollout servers with their own GPU pool, initialised via
    init_standalone() during start-up.  Never touched by add/remove_elastic_replica.

  Elastic hybrid replicas  (managed by ElasticAgentLoopManager)
    Share GPUs with the training engine (elastic_worker_group).
    RolloutReplica objects are created and initialised (init_hybrid) inside
    ElasticAgentLoopManager.create() from the elastic_worker_group returned by
    _get_elastic_worker_group().  After initialisation they are put to sleep.
    They start sleeping and are activated / deactivated on demand:
      add_elastic_replica(resource_id):    wake_up() → LB register
      remove_elastic_replica(resource_id): abort → LB cleanup → sleep()

Injection pattern
-----------------
Because ElasticRollouter is a Ray actor it cannot receive the elastic worker
group via the constructor easily (the worker group is created later by the
TaskRunner).  Use ``set_elastic_worker_group(wg)`` *before* calling
``init_workers()`` to inject the worker group, or pass it after construction
and call ``_reinit_elastic_replicas(wg)`` to register the hybrid replicas.
"""

import logging
from typing import TYPE_CHECKING, Optional

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ElasticRollouter(FullyAsyncRollouter):
    """
    Elastic Rollouter – thin wrapper over FullyAsyncRollouter.

    Added responsibilities
    ----------------------
    1. Swaps in ``ElasticAgentLoopManager`` (which understands how to
       dynamically add/remove rollout servers) in place of the standard
       ``FullyAsyncAgentLoopManager``.
    2. Holds a reference to the elastic worker group (``_elastic_worker_group``)
       so ``ElasticAgentLoopManager.create()`` can initialise hybrid replicas.
    3. Exposes ``set_elastic_worker_group()`` so the TaskRunner can inject the
       elastic worker group after construction.

    All elastic replica tracking, versioning, and statistics live in
    ``ElasticAgentLoopManager``.  ``ElasticRollouter`` merely forwards calls.

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
        # Injected by TaskRunner before init_workers() is called.
        self._elastic_worker_group: Optional[RayWorkerGroup] = None
        logger.info("[ElasticRollouter] Initialised (elastic server management via ElasticAgentLoopManager)")

    # -------------------------------------------------------------------------
    # Elastic worker group injection
    # -------------------------------------------------------------------------

    def set_elastic_worker_group(self, worker_group: RayWorkerGroup) -> None:
        """
        Inject the elastic worker group.

        Must be called **before** ``init_workers()`` so that
        ``_init_async_rollout_manager`` can pass it to
        ``ElasticAgentLoopManager.create()``.

        If ``init_workers()`` has already been called, use
        ``_reinit_elastic_replicas()`` instead to register the hybrid replicas
        after the fact.

        Args:
            worker_group: The RayWorkerGroup whose GPUs are shared with the
                training engine.  Each worker in this group will back one
                elastic hybrid rollout replica.
        """
        self._elastic_worker_group = worker_group
        logger.info(
            f"[ElasticRollouter] Elastic worker group set (world_size={getattr(worker_group, 'world_size', '?')})"
        )

    # -------------------------------------------------------------------------
    # Override: swap in ElasticAgentLoopManager
    # -------------------------------------------------------------------------

    async def _init_async_rollout_manager(self):
        """
        Use ElasticAgentLoopManager instead of FullyAsyncAgentLoopManager.

        Fixed standalone replicas are created by the base class initialisation
        (worker_group=self.rollout_wg → init_hybrid for each replica).

        Elastic hybrid replicas are created inside ElasticAgentLoopManager.create()
        from the elastic_worker_group set via set_elastic_worker_group().
        Subclasses that have elastic resources should call set_elastic_worker_group()
        before init_workers().
        """
        from verl.experimental.elastic_scheduling.agent_loop.elastic_agent_loop import ElasticAgentLoopManager

        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        assert self.config.actor_rollout_ref.rollout.mode == "async"

        self.async_rollout_mode = True
        elastic_wg = self._get_elastic_worker_group()
        self.async_rollout_manager: ElasticAgentLoopManager = await ElasticAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
            reward_loop_worker_handles=reward_loop_worker_handles,
            elastic_worker_group=elastic_wg,
        )
        logger.info(
            f"[ElasticRollouter] ElasticAgentLoopManager initialised with "
            f"{len(self.async_rollout_manager.rollout_replicas)} fixed replicas, "
            f"{len(self.async_rollout_manager._registered_elastic_replicas)} elastic replicas "
            f"registered (sleeping)"
        )

    def _get_elastic_worker_group(self) -> "RayWorkerGroup | None":
        """
        Return the worker group for elastic hybrid replicas.

        Returns the worker group injected via ``set_elastic_worker_group()``,
        or ``None`` if none was injected (no elastic replicas at start).
        """
        return self._elastic_worker_group

    # -------------------------------------------------------------------------
    # Elastic replica management – thin delegation to async_rollout_manager
    # -------------------------------------------------------------------------

    async def add_elastic_replica(self, resource_id: str, param_version: int) -> bool:
        """
        Activate a pre-registered elastic hybrid replica.

        The replica must have been supplied via ``_get_elastic_worker_group()``
        at initialisation time.  ElasticAgentLoopManager will wake_up() the
        server and register it with the load balancer.

        Also adjusts max_concurrent_samples to account for the new replica.
        """
        ok = await self.async_rollout_manager.add_elastic_replica(resource_id, param_version)
        if ok and self.max_concurrent_samples is not None:
            async with self.lock:
                self.max_concurrent_samples += 16  # ~16 slots per replica
        return ok

    async def remove_elastic_replica(self, resource_id: str) -> bool:
        """
        Deactivate an active elastic hybrid replica.

        ElasticAgentLoopManager will abort in-flight requests (triggering
        partial-rollout auto-resume on healthy servers), deregister from the
        LB, then sleep() the server to return GPUs to the training engine.
        The replica remains pre-registered and can be re-activated later.

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

    def get_elastic_replica(self, resource_id: str):
        """
        Return the RolloutReplica object for a registered elastic resource.

        Used by ElasticCheckpointManager to register hybrid replicas for
        parameter synchronisation.

        Returns:
            RolloutReplica if found in _registered_elastic_replicas, else None.
        """
        return self.async_rollout_manager._registered_elastic_replicas.get(resource_id)

    def get_all_elastic_replicas(self) -> dict:
        """
        Return all registered elastic replicas (sleeping + active).

        Returns:
            Dict[resource_id → RolloutReplica] from _registered_elastic_replicas.
        """
        return dict(self.async_rollout_manager._registered_elastic_replicas)

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
        """Metadata for all active elastic replicas."""
        return self.async_rollout_manager.get_elastic_replicas_info()

    def get_total_produced_samples(self) -> int:
        """Total samples produced (uses base class counter)."""
        return self.total_generated_samples
