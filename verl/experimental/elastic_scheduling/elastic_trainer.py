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
Elastic Trainer for VERL

Extends FullyAsyncTrainer with dynamic training DP management:
- Supports adding/removing training DP instances (elastic resources)
- Triggers DP group rebuild when elastic resources join/leave
- Coordinates parameter synchronization with elastic trainer workers
- Monitors consumption rate for coordinator decisions
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils import omega_conf_to_dataclass
from verl.utils.profiler import marked_timer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10)
class ElasticTrainer(FullyAsyncTrainer):
    """
    Elastic Trainer with dynamic DP management.

    Extends FullyAsyncTrainer to support:
    1. Dynamic addition of elastic training actors (elastic resources joining training)
    2. Dynamic removal of elastic training actors (elastic resources leaving training)
    3. Coordinated DP group rebuild across all training workers
    4. Consumption rate tracking for coordinator decisions

    The key difference from FullyAsyncTrainer:
    - Maintains a registry of elastic actor worker groups
    - Provides add_elastic_actor() and remove_elastic_actor() methods
    - Triggers coordinated DP rebuild when elastic resources change
    - Hooks into _fit_update_weights() for elastic sync triggering

    DP Rebuild Coordination:
        When elastic resources join training, ALL train workers must synchronize
        and rebuild their DP communication groups simultaneously. This is done via:
        1. ElasticCoordinator signals all train workers to prepare for rebuild
        2. All train workers + new elastic workers execute rebuild in sync
        3. After rebuild, training resumes with new DP size

    Architecture:
        ElasticTrainer (Ray actor, driver)
            ├── Fixed Actor Worker Group (actor_wg)
            └── Elastic Actor Worker Groups (elastic_actor_wgs)
                ↑
            Coordinator triggers add/remove
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

        # Elastic actor management
        # resource_id -> RayWorkerGroup for that elastic resource
        self._elastic_actor_wgs: dict[str, RayWorkerGroup] = {}
        # resource_id -> param_version when this actor joined
        self._elastic_actor_versions: dict[str, int] = {}
        # resource_id -> Ray actor handles for direct communication
        self._elastic_actor_handles: dict[str, list] = {}

        # DP rebuild coordination
        self._dp_rebuild_pending: bool = False
        self._dp_rebuild_lock = asyncio.Lock()
        self._pending_elastic_adds: list = []  # queue of (resource_id, wg) to add
        self._pending_elastic_removes: list = []  # queue of resource_ids to remove

        # Callback for coordinator
        self._on_dp_rebuild_complete: Optional[Callable] = None

        # Training state tracking for elastic coordination
        self._is_training_step = False
        self._current_mini_batch_idx: int = 0

        # Consumption statistics
        self._samples_consumed_since_last_report: int = 0
        self._last_report_time: float = time.time()
        self._consumption_rate_ema: Optional[float] = None
        self._consumption_ema_alpha: float = 0.3
        self._total_consumed_samples: int = 0  # Cumulative total for ElasticCoordinator

        # Elastic actor statistics
        self._total_elastic_adds: int = 0
        self._total_elastic_removes: int = 0
        self._total_dp_rebuilds: int = 0
        self._last_dp_rebuild_time: float = 0.0

        logger.info("[ElasticTrainer] Initialized with elastic DP support")

    # =========================================================================
    # Elastic Actor Management (called by ElasticCoordinator)
    # =========================================================================

    async def add_elastic_actor(
        self,
        resource_id: str,
        actor_worker_group: RayWorkerGroup,
        actor_handles: list,
        param_version: int,
    ) -> bool:
        """
        Add an elastic actor worker group to the training pool.

        This is called by ElasticCoordinator when an elastic resource
        switches from Rollout to Train mode.

        IMPORTANT: This triggers a DP group rebuild across ALL training workers.
        The rebuild is deferred to the next safe opportunity (mini-batch boundary
        or just before the next fit_step starts).

        Args:
            resource_id: Unique identifier for the elastic resource
            actor_worker_group: The RayWorkerGroup for this elastic actor
            actor_handles: Direct Ray actor handles for the workers
            param_version: Current parameter version (to verify sync)

        Returns:
            True if successfully registered, False otherwise
        """
        async with self._dp_rebuild_lock:
            if resource_id in self._elastic_actor_wgs:
                logger.warning(f"[ElasticTrainer] Elastic actor {resource_id} already registered")
                return False

            logger.info(
                f"[ElasticTrainer] Registering elastic actor {resource_id} "
                f"(param_version={param_version}). "
                f"Will rebuild DP on next safe opportunity."
            )

            # Register for pending rebuild
            self._elastic_actor_wgs[resource_id] = actor_worker_group
            self._elastic_actor_versions[resource_id] = param_version
            self._elastic_actor_handles[resource_id] = actor_handles
            self._pending_elastic_adds.append(resource_id)
            self._dp_rebuild_pending = True

            self._total_elastic_adds += 1

            return True

    async def remove_elastic_actor(self, resource_id: str) -> bool:
        """
        Remove an elastic actor worker group from the training pool.

        This is called by ElasticCoordinator when an elastic resource
        switches from Train to Rollout mode.

        Similar to add_elastic_actor(), the actual DP group rebuild
        is deferred to the next safe opportunity.

        Args:
            resource_id: Unique identifier for the elastic resource

        Returns:
            True if successfully queued for removal, False otherwise
        """
        async with self._dp_rebuild_lock:
            if resource_id not in self._elastic_actor_wgs:
                logger.warning(f"[ElasticTrainer] Elastic actor {resource_id} not found")
                return False

            logger.info(
                f"[ElasticTrainer] Queueing elastic actor {resource_id} for removal. "
                f"Will rebuild DP on next safe opportunity."
            )

            self._pending_elastic_removes.append(resource_id)
            self._dp_rebuild_pending = True

            self._total_elastic_removes += 1

            return True

    async def _apply_pending_dp_changes(self):
        """
        Apply pending DP group changes (add/remove elastic workers).

        This is called at mini-batch boundaries to ensure we don't interrupt
        an in-progress training step.

        The actual DP rebuild is executed on the worker processes via
        Ray remote calls to the actor worker groups.
        """
        if not self._dp_rebuild_pending:
            return

        async with self._dp_rebuild_lock:
            if not (self._pending_elastic_adds or self._pending_elastic_removes):
                self._dp_rebuild_pending = False
                return

            logger.info(
                f"[ElasticTrainer] Applying pending DP changes: "
                f"adds={self._pending_elastic_adds}, "
                f"removes={self._pending_elastic_removes}"
            )

            try:
                # Collect all current training world ranks via ElasticActorWorker.get_global_rank()
                current_dp_ranks = await self._get_current_train_ranks()

                # Compute new world ranks using worker-group-level rank queries
                add_ranks = []
                for rid in self._pending_elastic_adds:
                    if rid in self._elastic_actor_wgs:
                        ranks = await self._get_worker_group_ranks(self._elastic_actor_wgs[rid])
                        add_ranks.extend(ranks)

                remove_ranks = []
                for rid in self._pending_elastic_removes:
                    if rid in self._elastic_actor_wgs:
                        ranks = await self._get_worker_group_ranks(self._elastic_actor_wgs[rid])
                        remove_ranks.extend(ranks)

                new_world_ranks = [r for r in (current_dp_ranks + add_ranks) if r not in remove_ranks]

                # Execute coordinated DP rebuild
                await self._coordinate_dp_rebuild(
                    new_world_ranks=new_world_ranks,
                    add_resource_ids=list(self._pending_elastic_adds),
                    remove_resource_ids=list(self._pending_elastic_removes),
                )

                # Finalize removals
                for rid in self._pending_elastic_removes:
                    self._elastic_actor_wgs.pop(rid, None)
                    self._elastic_actor_versions.pop(rid, None)
                    self._elastic_actor_handles.pop(rid, None)

                # Clear pending lists
                self._pending_elastic_adds.clear()
                self._pending_elastic_removes.clear()
                self._dp_rebuild_pending = False

                self._total_dp_rebuilds += 1
                self._last_dp_rebuild_time = time.time()

                # Notify coordinator
                if self._on_dp_rebuild_complete:
                    await self._on_dp_rebuild_complete(len(new_world_ranks))

                logger.info(
                    f"[ElasticTrainer] DP changes applied successfully. "
                    f"New world size: {len(new_world_ranks)}, "
                    f"Total DP rebuilds: {self._total_dp_rebuilds}"
                )

            except Exception as e:
                logger.error(f"[ElasticTrainer] Failed to apply DP changes: {e}")
                # Don't clear pending lists - retry next opportunity
                raise

    async def _coordinate_dp_rebuild(
        self,
        new_world_ranks: list[int],
        add_resource_ids: list[str],
        remove_resource_ids: list[str],
    ):
        """
        Coordinate DP group rebuild across all training workers.

        Broadcasts rebuild_dp_group(new_world_ranks) to every worker group
        that participates in the new DP topology:
          1. Fixed actor worker group (actor_wg)
          2. Existing elastic actor groups that are NOT being removed
          3. New elastic worker groups being added

        All workers must call rebuild_dp_group concurrently because the
        underlying dist.new_group() is a collective that requires every
        current-world-size rank to participate simultaneously.

        The strategy-specific logic lives entirely in
        ElasticActorWorker.rebuild_dp_group(); the trainer is strategy-agnostic.

        Args:
            new_world_ranks: Complete list of ranks in new DP group
            add_resource_ids: Resource IDs being added
            remove_resource_ids: Resource IDs being removed
        """
        rebuild_futures = []

        # 1. Trigger rebuild on fixed actor workers
        if hasattr(self, "actor_wg") and self.actor_wg is not None:
            rebuild_futures.append(self._trigger_rebuild_on_worker_group(self.actor_wg, new_world_ranks))

        # 2. Trigger rebuild on existing elastic actor workers (not being removed)
        for rid, wg in self._elastic_actor_wgs.items():
            if rid not in remove_resource_ids:
                rebuild_futures.append(self._trigger_rebuild_on_worker_group(wg, new_world_ranks))

        # 3. Trigger rebuild on new elastic workers being added
        for rid in add_resource_ids:
            if rid in self._elastic_actor_wgs:
                rebuild_futures.append(
                    self._trigger_rebuild_on_worker_group(self._elastic_actor_wgs[rid], new_world_ranks)
                )

        # Execute all rebuilds concurrently
        if rebuild_futures:
            results = await asyncio.gather(*rebuild_futures, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[ElasticTrainer] DP rebuild failed for worker group {i}: {result}")
                    raise result

    async def _trigger_rebuild_on_worker_group(
        self,
        worker_group: RayWorkerGroup,
        new_world_ranks: list[int],
    ):
        """
        Trigger DP rebuild on a specific worker group.

        Calls ElasticActorWorker.rebuild_dp_group(new_world_ranks) on every
        worker in the group.  The strategy-specific logic (fsdp / fsdp2 /
        megatron) is fully encapsulated in ElasticActorWorker; the trainer
        does not need to know which strategy is in use.

        Args:
            worker_group: The RayWorkerGroup to broadcast the rebuild to.
            new_world_ranks: Complete, ordered list of global ranks that form
                the new data-parallel group.
        """
        try:
            futures = worker_group.execute_all(
                "rebuild_dp_group",
                new_world_ranks=new_world_ranks,
            )
            ray.get(futures)
        except Exception as e:
            logger.error(f"[ElasticTrainer] rebuild_dp_group failed on worker group: {e}")
            raise

    async def _get_current_train_ranks(self) -> list[int]:
        """
        Get global ranks of all current fixed training workers.

        Calls ElasticActorWorker.get_global_rank() on each worker in the
        fixed actor worker group.  Returns the sorted unique rank list.
        """
        if not (hasattr(self, "actor_wg") and self.actor_wg is not None):
            return []
        return await self._get_worker_group_ranks(self.actor_wg)

    async def _get_worker_group_ranks(self, worker_group: RayWorkerGroup) -> list[int]:
        """
        Query each worker in worker_group for its global rank.

        Calls ElasticActorWorker.get_global_rank() on all workers via
        worker_group.execute_all().  This replaces the old approach of
        holding Ray actor handles separately.

        Args:
            worker_group: The RayWorkerGroup to query.

        Returns:
            Sorted list of global ranks reported by each worker.
        """
        try:
            futures = worker_group.execute_all("get_global_rank")
            ranks = ray.get(futures)
            return sorted(set(ranks))
        except Exception as e:
            logger.warning(f"[ElasticTrainer] Could not get worker ranks: {e}")
            return []

    async def _get_worker_ranks(self, handles: list) -> list[int]:
        """
        Get global ranks from a list of elastic actor handles.

        Kept for backward compatibility; prefer _get_worker_group_ranks() for
        new code.  Calls get_global_rank.remote() on each handle directly.

        Args:
            handles: List of Ray actor handles (ElasticActorWorker instances).

        Returns:
            List of global ranks, one per handle.
        """
        ranks = []
        try:
            rank_futures = [handle.get_global_rank.remote() for handle in handles]
            ranks = await asyncio.gather(*[asyncio.wrap_future(f.future()) for f in rank_futures])
        except Exception as e:
            logger.warning(f"[ElasticTrainer] Could not get worker ranks: {e}")
        return list(ranks)

    # =========================================================================
    # Override Training Loop to Support Elastic DP Changes
    # =========================================================================

    async def fit_step(self, batch_dict: dict = None):
        """
        Extended fit_step with elastic role-switch at the training boundary.

        Flow:
        1. _fit_generate() — waits until required_samples are collected (Train GPU idle)
        2. [elastic hook] _elastic_on_before_fit_step() — coordinator executes any pending
           switch here while Train GPU is still idle:
               - sleep/wake_up role switch
               - DP rebuild on train workers
               - apply any pending elastic adds/removes
        3. _fit_compute_reward() / _fit_update_actor() ... — training resumes (GPU busy)
        4. _fit_update_weights() — parameter sync to rollout replicas

        This ordering guarantees the Train GPU is always idle during role switches,
        so the switch cost is paid while we were already waiting for data.
        """
        self._is_training_step = False
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}
        self._fit_start_profile()

        try:
            # --- Phase 1: collect data (Train GPU idle) ---
            batch = await self._fit_generate(None)

            # --- Phase 2: elastic switch (Train GPU still idle) ---
            await self._elastic_on_before_fit_step()
            if self._dp_rebuild_pending:
                try:
                    await self._apply_pending_dp_changes()
                except Exception as e:
                    logger.error(f"[ElasticTrainer] DP change application failed: {e}")
                    self._dp_rebuild_pending = False
                    self._pending_elastic_adds.clear()
                    self._pending_elastic_removes.clear()

            # --- Phase 3: training (Train GPU busy) ---
            self._is_training_step = True
            with marked_timer("step", self.timing_raw):
                batch = self._fit_compute_reward(batch)
                batch = self._fit_compute_log_prob(batch)
                batch = self._fit_compute_ref_log_prob(batch)
                batch = self._fit_compute_critic(batch)
                batch = self._fit_compute_advantage(batch)
                batch = self._fit_update_critic(batch)
                batch = self._fit_update_actor(batch)
                self._fit_update_local_step()
                await self._fit_update_weights()
                self._fit_dump_data(batch)

            await self._fit_validate()
            self._fit_save_checkpoint()
            self._fit_stop_profile()
            self._fit_collect_metrics(batch)
            self._fit_torch_memory()
            self._fit_postprocess_step()
        finally:
            self._is_training_step = False

    async def _elastic_on_before_fit_step(self):
        """
        Notify ElasticCoordinator that a training boundary has been reached.
        The coordinator will execute any pending role switch at this point,
        while Train GPU is guaranteed idle.
        """
        if not (hasattr(self, "_elastic_coordinator") and self._elastic_coordinator is not None):
            return
        try:
            switched = await asyncio.wrap_future(
                self._elastic_coordinator.on_before_fit_step.remote(step=self.global_steps).future()
            )
            if switched:
                logger.info(f"[ElasticTrainer] Elastic role switch completed at step={self.global_steps}")
        except Exception as e:
            logger.warning(f"[ElasticTrainer] on_before_fit_step hook failed: {e}")

    # =========================================================================
    # Extended Parameter Sync
    # =========================================================================

    def _setup_checkpoint_manager(self, rollouter):
        """
        Override to use ElasticCheckpointManager instead of CheckpointEngineManager.

        ElasticCheckpointManager inherits the full standalone-replica sync pipeline
        and additionally supports hybrid replicas (add_hybrid_replica /
        remove_hybrid_replica).  Both standalone and hybrid replicas are synced
        in a single update_weights() call.

        Args:
            rollouter: Ray actor handle to ElasticRollouter
        """
        from verl.experimental.elastic_scheduling.elastic_checkpoint_engine import ElasticCheckpointManager

        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = ElasticCheckpointManager(
            config=checkpoint_engine_config,
            trainer=self.actor_wg,
            replicas=replicas,
        )
        logger.info("[ElasticTrainer] ElasticCheckpointManager initialized")

    def register_hybrid_replicas(self, replicas: dict) -> None:
        """
        Register hybrid rollout replicas with the checkpoint manager.

        Call this once after initialisation to register all pre-bound elastic
        hybrid replicas (those that share GPUs with the training engine).  These
        replicas are registered **permanently** – they remain in the checkpoint
        manager regardless of their current sleep state so that every
        update_weights() call reaches all of them (sleeping replicas are woken up
        for the sync and then put back to sleep; awake replicas are handled via
        the normal abort → sync → resume flow).

        This method is idempotent: calling it multiple times for the same
        resource_id simply overwrites the previous entry.

        Args:
            replicas: Mapping of resource_id → RolloutReplica for all elastic
                hybrid replicas that should participate in parameter sync.
        """
        if not hasattr(self, "checkpoint_manager") or self.checkpoint_manager is None:
            logger.warning(
                "[ElasticTrainer] register_hybrid_replicas called before checkpoint_manager "
                "is ready (call set_rollouter first). Replicas not registered."
            )
            return

        from verl.experimental.elastic_scheduling.elastic_checkpoint_engine import ElasticCheckpointManager

        if not isinstance(self.checkpoint_manager, ElasticCheckpointManager):
            logger.warning(
                "[ElasticTrainer] checkpoint_manager is not an ElasticCheckpointManager; "
                "hybrid replicas cannot be registered."
            )
            return

        for resource_id, replica in replicas.items():
            self.checkpoint_manager.add_hybrid_replica(resource_id, replica)

        logger.info(f"[ElasticTrainer] Registered {len(replicas)} hybrid replica(s) with checkpoint manager")

    def set_elastic_coordinator(self, coordinator):
        """
        Set the ElasticCoordinator for pre-sync hook integration.

        Args:
            coordinator: Ray actor handle to ElasticCoordinator
        """
        self._elastic_coordinator = coordinator
        logger.info("[ElasticTrainer] ElasticCoordinator reference set")

    async def _fit_update_weights(self):
        """
        Sync parameters from trainer to all rollout replicas.

        Role switches happen earlier (in _elastic_on_before_fit_step), so by
        the time this is called the replica membership is stable.

        ElasticCheckpointManager.update_weights() handles both:
          - Standalone replicas (self.checkpoint_manager.replicas)
          - Hybrid replicas (registered via add_hybrid_replica / remove_hybrid_replica),
            including those currently sleeping and those currently awake.
        """
        await super()._fit_update_weights()

    # =========================================================================
    # Consumption Rate Tracking
    # =========================================================================

    def _fit_postprocess_step(self):
        """Extended postprocess to track consumption rate."""
        super()._fit_postprocess_step()

        # Update consumption tracking
        n_consumed = getattr(self, "required_samples", 0)
        self._samples_consumed_since_last_report += n_consumed
        self._total_consumed_samples += n_consumed

        # Update EMA of consumption rate
        current_time = time.time()
        elapsed = current_time - self._last_report_time
        if elapsed > 0:
            instant_rate = self._samples_consumed_since_last_report / elapsed
            if self._consumption_rate_ema is None:
                self._consumption_rate_ema = instant_rate
            else:
                self._consumption_rate_ema = (
                    self._consumption_ema_alpha * instant_rate
                    + (1 - self._consumption_ema_alpha) * self._consumption_rate_ema
                )
            self._last_report_time = current_time
            self._samples_consumed_since_last_report = 0

    def get_consumption_rate(self) -> float:
        """Get current consumption rate (samples/sec) for monitoring."""
        return self._consumption_rate_ema or 0.0

    def get_total_consumed_samples(self) -> int:
        """
        Get total number of samples consumed since start.

        Used by ElasticCoordinator to compute consumption rate.
        """
        return self._total_consumed_samples

    def get_elastic_statistics(self) -> dict:
        """Get elastic-specific statistics for monitoring."""
        return {
            "elastic_trainer/num_elastic_actors": len(self._elastic_actor_wgs),
            "elastic_trainer/total_adds": self._total_elastic_adds,
            "elastic_trainer/total_removes": self._total_elastic_removes,
            "elastic_trainer/total_dp_rebuilds": self._total_dp_rebuilds,
            "elastic_trainer/last_dp_rebuild_time": self._last_dp_rebuild_time,
            "elastic_trainer/consumption_rate_ema": self._consumption_rate_ema or 0.0,
            "elastic_trainer/dp_rebuild_pending": self._dp_rebuild_pending,
        }

    def get_num_active_train_actors(self) -> int:
        """Get total number of active training actors (fixed + elastic)."""
        fixed = 1 if (hasattr(self, "actor_wg") and self.actor_wg is not None) else 0
        return fixed + len(self._elastic_actor_wgs)

    def set_dp_rebuild_complete_callback(self, callback: Callable):
        """Set callback to be called after DP rebuild completes."""
        self._on_dp_rebuild_complete = callback
