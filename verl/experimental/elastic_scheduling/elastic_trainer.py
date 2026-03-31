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

Extends FullyAsyncTrainer with dynamic training DP management and elastic
resource role switching.

Switch sequence ownership
-------------------------
ElasticTrainer owns the **complete** switch sequence for both directions.
ElasticCoordinator only calls two high-level methods:

    switch_elastic_to_rollout(resource_id, param_version)
    switch_elastic_to_train(resource_id, param_version)

Train → Rollout  (switch_elastic_to_rollout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. remove_elastic_actor()         ← rebuild DP without this rank
2. [via ElasticActorWorker wg]    worker.switch_to_rollout()
                                  ← offload actor weights to CPU
3. [via rollouter]                rollouter.add_elastic_replica()
                                  ← wake_up rollout server + LB register

Rollout → Train  (switch_elastic_to_train)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. [via rollouter]                rollouter.remove_elastic_replica()
                                  ← sleep rollout server + abort in-flight
2. [via ElasticActorWorker wg]    worker.switch_to_train()
                                  ← load weights to GPU
3. add_elastic_actor()            ← rebuild DP with this rank

GPU memory is always available at each step because:
- In step 1 (rollout→train): rollout server sleeps before actor loads.
- In step 2 (train→rollout): actor offloads before rollout server wakes.
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


class ElasticTrainer(FullyAsyncTrainer):
    """
    Elastic Trainer with dynamic DP management and role switching.

    New public API (called by ElasticCoordinator)
    -----------------------------------------------
    switch_elastic_to_rollout(resource_id, param_version) → bool
        Complete Train→Rollout sequence.

    switch_elastic_to_train(resource_id, param_version) → bool
        Complete Rollout→Train sequence.

    Internal elastic actor registry
    ---------------------------------
    _elastic_actor_wgs: resource_id → RayWorkerGroup
        Worker groups currently in Train mode (for DP rebuild).

    _pending_elastic_adds / _pending_elastic_removes
        DP changes deferred to mini-batch boundaries.
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

        # Elastic actor registry: resource_id → RayWorkerGroup (train mode)
        self._elastic_actor_wgs: dict[str, RayWorkerGroup] = {}
        # resource_id → param_version when this actor joined training
        self._elastic_actor_versions: dict[str, int] = {}

        # DP rebuild coordination
        self._dp_rebuild_pending: bool = False
        self._dp_rebuild_lock = asyncio.Lock()
        self._pending_elastic_adds: list[str] = []
        self._pending_elastic_removes: list[str] = []

        # Callback for external notification after rebuild
        self._on_dp_rebuild_complete: Optional[Callable] = None

        # Training state
        self._is_training_step = False
        self._current_mini_batch_idx: int = 0

        # Consumption statistics
        self._samples_consumed_since_last_report: int = 0
        self._last_report_time: float = time.time()
        self._consumption_rate_ema: Optional[float] = None
        self._consumption_ema_alpha: float = 0.3
        self._total_consumed_samples: int = 0

        # Elastic statistics
        self._total_elastic_adds: int = 0
        self._total_elastic_removes: int = 0
        self._total_dp_rebuilds: int = 0
        self._last_dp_rebuild_time: float = 0.0
        # Latency of the most recent complete role switch (seconds, 0 before first switch)
        self._last_switch_latency: float = 0.0
        self._switch_start_time: float = 0.0

        # Coordinator reference (optional, for pre-sync hook)
        self._elastic_coordinator = None

        logger.info("[ElasticTrainer] Initialized with elastic DP support")

    # =========================================================================
    # Worker Initialization Overrides
    # =========================================================================

    def _is_fully_elastic(self) -> bool:
        """Return True when trainer.n_gpus_per_node == 0 (all-elastic mode)."""
        return int(getattr(self.config.trainer, "n_gpus_per_node", 0)) == 0

    def _create_actor_rollout_classes(self):
        """
        In fully-elastic mode (trainer.n_gpus_per_node=0), there is no fixed
        training resource pool, so skip creating actor rollout classes here.
        The elastic worker groups are wired in separately via
        register_elastic_worker_group().

        In normal mode, delegate to the parent implementation.
        """
        if self._is_fully_elastic():
            logger.info("[ElasticTrainer] Fully-elastic mode: skipping fixed actor rollout class creation")
            return
        super()._create_actor_rollout_classes()

    def _init_models(self):
        """
        In fully-elastic mode, skip actor_wg initialization (no fixed actor
        resource pool).  Critic / ref-policy / reward-model are still
        initialised as usual if present.
        """
        if self._is_fully_elastic():
            logger.info("[ElasticTrainer] Fully-elastic mode: skipping fixed actor_wg init_model")
            # Only initialise non-actor worker groups
            if self.use_critic:
                from verl.trainer.ppo.utils import Role as _Role

                self.critic_wg = self.all_wg[str(_Role.Critic)]
                self.critic_wg.init_model()

            if self.use_reference_policy and not self.ref_in_actor:
                from verl.trainer.ppo.utils import Role as _Role

                self.ref_policy_wg = self.all_wg[str(_Role.RefPolicy)]
                self.ref_policy_wg.init_model()

            if self.use_rm:
                from verl.trainer.ppo.utils import Role as _Role

                self.rm_wg = self.all_wg[str(_Role.RewardModel)]
                self.rm_wg.init_model()

            # actor_wg / actor_rollout_wg will be set to the first elastic wg
            # in register_elastic_worker_group() once elastic wgs are created.
            self.actor_wg = None
            self.actor_rollout_wg = None
            return

        super()._init_models()

    # =========================================================================
    # Complete Switch Sequences (called by ElasticCoordinator)
    # =========================================================================

    async def switch_elastic_to_rollout(self, resource_id: str, param_version: int) -> bool:
        """
        Complete Train → Rollout switch for one elastic resource.

        Sequence:
        1. remove_elastic_actor()   – rebuild DP without this rank
        2. worker.switch_to_rollout() – offload actor weights to CPU
        3. rollouter.add_elastic_replica() – wake rollout server + LB register

        Returns:
            True if all three steps succeeded, False otherwise.
        """
        logger.info(f"[ElasticTrainer] switch_elastic_to_rollout: {resource_id}")
        _t0 = time.time()

        if resource_id not in self._elastic_actor_wgs:
            logger.warning(f"[ElasticTrainer] {resource_id} not in elastic actor registry")
            return False

        wg = self._elastic_actor_wgs[resource_id]

        try:
            # Step 1: Remove from training DP group (rebuild DP without this rank)
            await self.remove_elastic_actor(resource_id)

            # Step 2: Offload actor weights to CPU on the worker
            # (GPU memory freed for rollout server)
            try:
                futures = wg.execute_all("switch_to_rollout", param_version=param_version)
                results = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
                if not all(results):
                    logger.warning(f"[ElasticTrainer] Some workers failed switch_to_rollout for {resource_id}")
            except Exception as e:
                logger.warning(f"[ElasticTrainer] switch_to_rollout on worker failed: {e}")

            # Step 3: Wake up the rollout server via rollouter
            if self.rollouter is not None:
                try:
                    ok = await asyncio.wrap_future(
                        self.rollouter.add_elastic_replica.remote(
                            resource_id=resource_id,
                            param_version=param_version,
                        ).future()
                    )
                    if not ok:
                        logger.warning(f"[ElasticTrainer] rollouter.add_elastic_replica failed for {resource_id}")
                    else:
                        # Update cached active rollout replica count
                        self._elastic_rollout_replicas_cache = list(
                            getattr(self, "_elastic_rollout_replicas_cache", [])
                        ) + [resource_id]
                        # Mark the hybrid replica as AWAKE so that the next
                        # update_weights() uses NCCL sync instead of in-process sync.
                        if hasattr(self, "checkpoint_manager") and self.checkpoint_manager is not None:
                            self.checkpoint_manager.mark_hybrid_awake([resource_id])
                except Exception as e:
                    logger.error(f"[ElasticTrainer] Failed to add elastic replica for {resource_id}: {e}")
                    return False

            self._last_switch_latency = time.time() - _t0
            logger.info(f"[ElasticTrainer] {resource_id} switched to rollout mode in {self._last_switch_latency:.2f}s")
            return True

        except Exception as e:
            logger.error(f"[ElasticTrainer] switch_elastic_to_rollout failed for {resource_id}: {e}")
            return False

    async def switch_elastic_to_train(self, resource_id: str, param_version: int) -> bool:
        """
        Complete Rollout → Train switch for one elastic resource.

        Sequence:
        1. rollouter.remove_elastic_replica() – sleep rollout server + abort in-flight
        2. worker.switch_to_train() – load weights to GPU
        3. add_elastic_actor() – rebuild DP with this rank

        ``resource_id`` must correspond to a worker group that was registered
        at initialisation time (via ``register_elastic_worker_group``).

        Returns:
            True if all three steps succeeded, False otherwise.
        """
        logger.info(f"[ElasticTrainer] switch_elastic_to_train: {resource_id}")
        _t0 = time.time()

        wg = self._get_elastic_wg_by_resource_id(resource_id)
        if wg is None:
            logger.warning(
                f"[ElasticTrainer] No worker group registered for {resource_id}. "
                "Call register_elastic_worker_group() at init time."
            )
            return False

        try:
            # Step 1: Sleep the rollout server (releases GPU for actor model)
            if self.rollouter is not None:
                try:
                    ok = await asyncio.wrap_future(
                        self.rollouter.remove_elastic_replica.remote(
                            resource_id=resource_id,
                        ).future()
                    )
                    if not ok:
                        logger.warning(f"[ElasticTrainer] rollouter.remove_elastic_replica failed for {resource_id}")
                    else:
                        # Update cached active rollout replica count
                        cache = list(getattr(self, "_elastic_rollout_replicas_cache", []))
                        if resource_id in cache:
                            cache.remove(resource_id)
                        self._elastic_rollout_replicas_cache = cache
                        # Mark the hybrid replica as SLEEPING so that the next
                        # update_weights() uses in-process sync instead of NCCL.
                        if hasattr(self, "checkpoint_manager") and self.checkpoint_manager is not None:
                            self.checkpoint_manager.mark_hybrid_sleeping([resource_id])
                except Exception as e:
                    logger.warning(f"[ElasticTrainer] Failed to remove elastic replica for {resource_id}: {e}")

            # Step 2: Compute the new train world ranks
            # (current fixed ranks + this resource's ranks)
            current_ranks = await self._get_current_train_ranks()
            new_ranks = await self._get_worker_group_ranks(wg)
            new_train_world_ranks = sorted(set(current_ranks) | set(new_ranks))

            # Load actor weights to GPU on the worker
            try:
                futures = wg.execute_all(
                    "switch_to_train",
                    new_train_world_ranks=new_train_world_ranks,
                    param_version=param_version,
                )
                results = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
                if not all(results):
                    logger.warning(f"[ElasticTrainer] Some workers failed switch_to_train for {resource_id}")
            except Exception as e:
                logger.warning(f"[ElasticTrainer] switch_to_train on worker failed: {e}")

            # Step 3: Add to training DP group (rebuild DP with this rank)
            await self.add_elastic_actor(
                resource_id=resource_id,
                actor_worker_group=wg,
                actor_handles=[],  # handles not needed; wg is sufficient
                param_version=param_version,
            )

            self._last_switch_latency = time.time() - _t0
            logger.info(f"[ElasticTrainer] {resource_id} switched to train mode in {self._last_switch_latency:.2f}s")
            return True

        except Exception as e:
            logger.error(f"[ElasticTrainer] switch_elastic_to_train failed for {resource_id}: {e}")
            return False

    # =========================================================================
    # Elastic Worker Group Registry
    # =========================================================================

    def register_elastic_worker_group(self, resource_id: str, worker_group: RayWorkerGroup) -> None:
        """
        Pre-register an elastic worker group by resource_id.

        Must be called at init time (before any switch) so that
        switch_elastic_to_train() can look up the worker group when the
        resource switches from rollout to train mode.

        Args:
            resource_id: Unique identifier (e.g. "elastic_0").
            worker_group: The RayWorkerGroup wrapping the elastic workers.
        """
        self._elastic_wg_registry = getattr(self, "_elastic_wg_registry", {})
        self._elastic_wg_registry[resource_id] = worker_group

        # In fully-elastic mode (no fixed actor pool), promote the first
        # registered elastic wg to actor_wg so that the rest of the trainer
        # (checkpoint manager, param-sync, etc.) has a valid handle.
        if self._is_fully_elastic() and getattr(self, "actor_wg", None) is None:
            self.actor_wg = worker_group
            self.actor_rollout_wg = worker_group
            logger.info(f"[ElasticTrainer] Fully-elastic mode: promoted '{resource_id}' as actor_wg")

        logger.info(f"[ElasticTrainer] Registered elastic worker group: {resource_id}")

    def _get_elastic_wg_by_resource_id(self, resource_id: str) -> Optional[RayWorkerGroup]:
        """Look up the pre-registered worker group for a resource."""
        registry = getattr(self, "_elastic_wg_registry", {})
        return registry.get(resource_id)

    # =========================================================================
    # Elastic Actor Management (lower-level, used internally)
    # =========================================================================

    async def add_elastic_actor(
        self,
        resource_id: str,
        actor_worker_group: RayWorkerGroup,
        actor_handles: list,
        param_version: int,
    ) -> bool:
        """
        Add an elastic actor worker group to the training pool and trigger
        a DP group rebuild.

        Called by switch_elastic_to_train() after the actor weights are on GPU.

        Args:
            resource_id: Unique identifier for the elastic resource.
            actor_worker_group: The RayWorkerGroup for this elastic actor.
            actor_handles: (unused) kept for API compatibility.
            param_version: Current parameter version.

        Returns:
            True if successfully registered, False otherwise.
        """
        async with self._dp_rebuild_lock:
            if resource_id in self._elastic_actor_wgs:
                logger.warning(f"[ElasticTrainer] {resource_id} already registered")
                return False

            self._elastic_actor_wgs[resource_id] = actor_worker_group
            self._elastic_actor_versions[resource_id] = param_version
            self._pending_elastic_adds.append(resource_id)
            self._dp_rebuild_pending = True
            self._total_elastic_adds += 1

            logger.info(
                f"[ElasticTrainer] Registered elastic actor {resource_id} "
                f"(param_version={param_version}). DP rebuild deferred."
            )

        # Apply the pending DP change immediately if we're not mid-step
        if not self._is_training_step:
            await self._apply_pending_dp_changes()

        return True

    async def remove_elastic_actor(self, resource_id: str) -> bool:
        """
        Remove an elastic actor worker group from the training pool and trigger
        a DP group rebuild.

        Called by switch_elastic_to_rollout() before the actor weights are
        offloaded.

        Args:
            resource_id: Unique identifier for the elastic resource.

        Returns:
            True if successfully queued for removal, False otherwise.
        """
        async with self._dp_rebuild_lock:
            if resource_id not in self._elastic_actor_wgs:
                logger.warning(f"[ElasticTrainer] {resource_id} not found for removal")
                return False

            self._pending_elastic_removes.append(resource_id)
            self._dp_rebuild_pending = True
            self._total_elastic_removes += 1

            logger.info(f"[ElasticTrainer] Queued elastic actor {resource_id} for removal. DP rebuild deferred.")

        # Apply immediately if not mid-step
        if not self._is_training_step:
            await self._apply_pending_dp_changes()

        return True

    async def _apply_pending_dp_changes(self):
        """
        Apply pending DP group changes (add/remove elastic workers).

        Called at mini-batch boundaries or immediately after register/deregister
        when not mid-step.
        """
        if not self._dp_rebuild_pending:
            return

        async with self._dp_rebuild_lock:
            if not (self._pending_elastic_adds or self._pending_elastic_removes):
                self._dp_rebuild_pending = False
                return

            logger.info(
                f"[ElasticTrainer] Applying pending DP changes: "
                f"adds={self._pending_elastic_adds}, removes={self._pending_elastic_removes}"
            )

            try:
                current_dp_ranks = await self._get_current_train_ranks()

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

                new_world_ranks = sorted(set(current_dp_ranks + add_ranks) - set(remove_ranks))

                await self._coordinate_dp_rebuild(
                    new_world_ranks=new_world_ranks,
                    add_resource_ids=list(self._pending_elastic_adds),
                    remove_resource_ids=list(self._pending_elastic_removes),
                )

                # Finalise removals
                for rid in self._pending_elastic_removes:
                    self._elastic_actor_wgs.pop(rid, None)
                    self._elastic_actor_versions.pop(rid, None)

                self._pending_elastic_adds.clear()
                self._pending_elastic_removes.clear()
                self._dp_rebuild_pending = False

                self._total_dp_rebuilds += 1
                self._last_dp_rebuild_time = time.time()

                # 7.4: update required_samples to match new DP size
                self._update_required_samples()

                if self._on_dp_rebuild_complete:
                    await self._on_dp_rebuild_complete(len(new_world_ranks))

                logger.info(
                    f"[ElasticTrainer] DP changes applied. "
                    f"New world size: {len(new_world_ranks)}, "
                    f"required_samples updated to {self.required_samples}, "
                    f"Total DP rebuilds: {self._total_dp_rebuilds}"
                )

            except Exception as e:
                logger.error(f"[ElasticTrainer] Failed to apply DP changes: {e}")
                raise

    async def _coordinate_dp_rebuild(
        self,
        new_world_ranks: list[int],
        add_resource_ids: list[str],
        remove_resource_ids: list[str],
    ):
        """
        Broadcast rebuild_dp_group(new_world_ranks) to every worker group that
        participates in the new DP topology.

        All current-world-size ranks must participate in dist.new_group()
        simultaneously.
        """
        rebuild_futures = []

        # Fixed actor workers
        if hasattr(self, "actor_wg") and self.actor_wg is not None:
            rebuild_futures.append(self._trigger_rebuild_on_worker_group(self.actor_wg, new_world_ranks))

        # Existing elastic actors not being removed
        for rid, wg in self._elastic_actor_wgs.items():
            if rid not in remove_resource_ids:
                rebuild_futures.append(self._trigger_rebuild_on_worker_group(wg, new_world_ranks))

        # New elastic actors being added
        for rid in add_resource_ids:
            if rid in self._elastic_actor_wgs:
                rebuild_futures.append(
                    self._trigger_rebuild_on_worker_group(self._elastic_actor_wgs[rid], new_world_ranks)
                )

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
        """Call rebuild_dp_group on every worker in the group."""
        try:
            futures = worker_group.execute_all("rebuild_dp_group", new_world_ranks=new_world_ranks)
            await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
        except Exception as e:
            logger.error(f"[ElasticTrainer] rebuild_dp_group failed: {e}")
            raise

    # =========================================================================
    # 7.4  Dynamic required_samples (DP-aware)
    # =========================================================================

    def _get_current_dp_size(self) -> int:
        """
        Return the number of actor workers currently participating in training.

        Counts:
        - fixed actor_wg (world_size / tp_size / pp_size, i.e. dp_size contribution)
        - each elastic actor_wg in TRAIN mode (same calculation)

        We use world_size as a proxy for the DP contribution because all actor
        worker groups use the same TP/PP topology.  Dividing by tp×pp gives the
        DP size, but for the purpose of required_samples we only need the total
        number of data-parallel replicas across all wgs.
        """
        tp = getattr(self.config.actor_rollout_ref.actor, "tensor_model_parallel_size", 1)
        pp = getattr(self.config.actor_rollout_ref.actor, "pipeline_model_parallel_size", 1)
        parallelism = tp * pp
        parallelism = max(parallelism, 1)  # guard against zero

        dp_size = 0
        if hasattr(self, "actor_wg") and self.actor_wg is not None:
            dp_size += self.actor_wg.world_size // parallelism
        for wg in self._elastic_actor_wgs.values():
            dp_size += wg.world_size // parallelism
        return max(dp_size, 1)  # at least 1 to avoid division-by-zero

    def _update_required_samples(self) -> None:
        """
        Recompute required_samples so that it is always a multiple of both
        ppo_mini_batch_size and the current DP size.

        Formula::

            dp_size         = current number of DP replicas
            required_samples = ppo_mini_batch_size × require_batches × dp_size

        This guarantees that every DP rank receives exactly
        ``ppo_mini_batch_size × require_batches`` samples per step, avoiding
        shape mismatches that arise when sample counts are not evenly divisible
        by the data-parallel degree.

        Called automatically at the end of ``_apply_pending_dp_changes()``.
        """
        mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        require_batches = self.config.async_training.require_batches
        dp_size = self._get_current_dp_size()
        new_required = mini_batch_size * require_batches * dp_size
        old_required = self.required_samples
        self.required_samples = new_required
        logger.info(
            f"[ElasticTrainer] required_samples updated: {old_required} → {new_required} "
            f"(dp_size={dp_size}, mini_batch={mini_batch_size}, require_batches={require_batches})"
        )

    async def _get_current_train_ranks(self) -> list[int]:
        if not (hasattr(self, "actor_wg") and self.actor_wg is not None):
            return []
        return await self._get_worker_group_ranks(self.actor_wg)

    async def _get_worker_group_ranks(self, worker_group: RayWorkerGroup) -> list[int]:
        try:
            futures = worker_group.execute_all("get_global_rank")
            ranks = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
            return sorted(set(ranks))
        except Exception as e:
            logger.warning(f"[ElasticTrainer] Could not get worker ranks: {e}")
            return []

    # =========================================================================
    # Override Training Loop
    # =========================================================================

    async def fit_step(self, batch_dict: dict = None):
        """
        Extended fit_step with elastic role-switch hook at training boundary.

        Flow:
        1. _fit_generate() — collect data (Train GPU idle)
        2. _elastic_on_before_fit_step() — coordinator executes pending switch
           (Train GPU still idle; this is where role switches happen)
        3. _apply_pending_dp_changes() — apply any deferred DP changes
        4. Training compute (GPU busy)
        5. _fit_update_weights() — parameter sync to rollout replicas
        """
        self._is_training_step = False
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}
        self._fit_start_profile()

        try:
            # Phase 1: collect data (Train GPU idle)
            batch = await self._fit_generate(None)

            # Phase 2: elastic switch hook (Train GPU still idle)
            await self._elastic_on_before_fit_step()

            # Phase 3: apply any deferred DP changes
            if self._dp_rebuild_pending:
                try:
                    await self._apply_pending_dp_changes()
                except Exception as e:
                    logger.error(f"[ElasticTrainer] DP change application failed: {e}")
                    self._dp_rebuild_pending = False
                    self._pending_elastic_adds.clear()
                    self._pending_elastic_removes.clear()

            # Phase 4: training compute (GPU busy)
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
        Hook called when Train GPU is idle.  Asks the coordinator to execute
        any pending role switch at this safe boundary.
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
    # Parameter Sync Override
    # =========================================================================

    def _setup_checkpoint_manager(self, rollouter):
        """
        Override to use ElasticCheckpointManager instead of CheckpointEngineManager.

        ElasticCheckpointManager syncs both standalone replicas (fixed rollout
        servers) and hybrid replicas (elastic servers, sleeping or awake) in a
        single update_weights() call.
        """
        from verl.experimental.elastic_scheduling.elastic_checkpoint_engine import ElasticCheckpointManager

        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = ElasticCheckpointManager(
            config=checkpoint_engine_config,
            trainer=self.actor_wg,
            replicas=replicas,
        )
        # Cache the count of fixed (standalone) rollout replicas for metrics.
        # replicas is a list[RolloutReplica] returned by rollouter.get_replicas().
        # Elastic rollout replica counts are tracked separately via
        # _elastic_rollout_replicas_cache, updated by switch_elastic_to_*.
        self._fixed_rollout_replicas_cache = list(range(len(replicas))) if replicas else []
        self._elastic_rollout_replicas_cache = []
        logger.info("[ElasticTrainer] ElasticCheckpointManager initialized")

    def register_hybrid_replicas(self, replicas: dict) -> None:
        """
        Register hybrid rollout replicas with the checkpoint manager.

        Call this once after initialisation to register all pre-bound elastic
        hybrid replicas.  These replicas remain in the checkpoint manager
        regardless of their current sleep state.

        For each hybrid replica, the corresponding ElasticActorWorker worker
        group is looked up from ``_elastic_wg_registry`` (populated by
        ``register_elastic_worker_group()``) and passed to
        ``ElasticCheckpointManager.add_hybrid_replicas()`` so that the
        checkpoint manager can perform in-process weight sync when the replica
        is in TRAIN mode (sleeping).

        Args:
            replicas: Dict[resource_id → RolloutReplica].
        """
        from verl.experimental.elastic_scheduling.elastic_checkpoint_engine import ElasticCheckpointManager

        if not hasattr(self, "checkpoint_manager") or self.checkpoint_manager is None:
            logger.warning("[ElasticTrainer] register_hybrid_replicas called before checkpoint_manager is ready")
            return

        if not isinstance(self.checkpoint_manager, ElasticCheckpointManager):
            logger.warning("[ElasticTrainer] checkpoint_manager is not an ElasticCheckpointManager")
            return

        elastic_wg_registry = getattr(self, "_elastic_wg_registry", {})

        # A single elastic worker group (e.g. "elastic_0") may be sliced by
        # the rollouter into multiple DP replicas named "elastic_0", "elastic_1",
        # "elastic_2", etc.  The registry only holds one entry per physical wg
        # (keyed by the wg's own resource_id, typically "elastic_0").
        # For replicas that have no direct key match we fall back to the first
        # registry entry that shares the same group prefix (e.g. "elastic_").
        def _lookup_wg(rid: str):
            if rid in elastic_wg_registry:
                return elastic_wg_registry[rid]
            # Extract group prefix: "elastic_1" → "elastic_"
            prefix = rid.rsplit("_", 1)[0] + "_" if "_" in rid else rid
            for key, wg in elastic_wg_registry.items():
                if key.startswith(prefix):
                    return wg
            return None

        actor_wgs = {}
        missing = []
        for rid in replicas:
            wg = _lookup_wg(rid)
            if wg is not None:
                actor_wgs[rid] = wg
            else:
                missing.append(rid)

        for rid in missing:
            logger.warning(
                f"[ElasticTrainer] No actor_wg found for hybrid replica '{rid}'. "
                "In-process sync (TRAIN mode) will be skipped. "
                "Make sure register_elastic_worker_group() is called before register_hybrid_replicas()."
            )
        self.checkpoint_manager.add_hybrid_replicas(replicas, actor_wgs=actor_wgs)

        logger.info(f"[ElasticTrainer] Registered {len(replicas)} hybrid replica(s) with checkpoint manager")

    async def _fit_update_weights(self):
        """
        Sync parameters from trainer to ALL rollout replicas (standalone + hybrid).

        Role switches happen earlier (_elastic_on_before_fit_step), so replica
        membership is stable by the time this is called.
        """
        await super()._fit_update_weights()

    # =========================================================================
    # Coordinator wiring
    # =========================================================================

    def set_elastic_coordinator(self, coordinator) -> None:
        """Wire the ElasticCoordinator for pre-sync hook integration."""
        self._elastic_coordinator = coordinator
        logger.info("[ElasticTrainer] ElasticCoordinator reference set")

    def set_dp_rebuild_complete_callback(self, callback: Callable) -> None:
        """Set callback invoked after each DP rebuild."""
        self._on_dp_rebuild_complete = callback

    # =========================================================================
    # Consumption Rate Tracking
    # =========================================================================

    def _fit_postprocess_step(self):
        """Extended postprocess: track consumption rate and report elastic metrics."""
        super()._fit_postprocess_step()

        n_consumed = getattr(self, "required_samples", 0)
        self._samples_consumed_since_last_report += n_consumed
        self._total_consumed_samples += n_consumed

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

        # 7.5 Elastic metrics – written into self.metrics so they are picked
        # up by metrics_aggregator.add_step_metrics() in the base class.
        # num_rollout_replicas requires access to the rollouter; query it
        # asynchronously only if already available (avoid blocking here).
        num_fixed_replicas = len(getattr(self, "_fixed_rollout_replicas_cache", []))
        num_elastic_replicas = len(getattr(self, "_elastic_rollout_replicas_cache", []))
        num_rollout = num_fixed_replicas + num_elastic_replicas

        self.metrics.update(
            {
                "elastic/total_switch_to_rollout": self._total_elastic_removes,
                "elastic/total_switch_to_train": self._total_elastic_adds,
                "elastic/last_switch_latency_s": self._last_switch_latency,
                "elastic/num_train_actors": self.get_num_active_train_actors(),
                "elastic/current_dp_size": self._get_current_dp_size(),
                "elastic/required_samples": self.required_samples,
            }
        )
        if num_rollout > 0:
            self.metrics["elastic/num_rollout_replicas"] = num_rollout

    def get_consumption_rate(self) -> float:
        """Current consumption rate (samples/sec)."""
        return self._consumption_rate_ema or 0.0

    def get_total_consumed_samples(self) -> int:
        """Total samples consumed since start (used by ElasticCoordinator)."""
        return self._total_consumed_samples

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_num_active_train_actors(self) -> int:
        """Total number of active training actors (fixed + elastic)."""
        fixed = 1 if (hasattr(self, "actor_wg") and self.actor_wg is not None) else 0
        return fixed + len(self._elastic_actor_wgs)

    def get_elastic_statistics(self) -> dict:
        """Elastic-specific statistics for monitoring."""
        return {
            "elastic_trainer/num_elastic_actors": len(self._elastic_actor_wgs),
            "elastic_trainer/total_adds": self._total_elastic_adds,
            "elastic_trainer/total_removes": self._total_elastic_removes,
            "elastic_trainer/total_dp_rebuilds": self._total_dp_rebuilds,
            "elastic_trainer/last_dp_rebuild_time": self._last_dp_rebuild_time,
            "elastic_trainer/consumption_rate_ema": self._consumption_rate_ema or 0.0,
            "elastic_trainer/dp_rebuild_pending": self._dp_rebuild_pending,
        }
