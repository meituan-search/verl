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

    Internal elastic registries (single-wg architecture)
    ------------------------------------------------------
    _elastic_wg_registry: {"elastic_wg": RayWorkerGroup}
        Single entry holding the unified large RayWorkerGroup.
    _elastic_unit_ranks: resource_id → list[int]
        Maps each elastic unit id to its wg-local ranks (TP×PP×CP ranks).
    _elastic_active_units: set[str]
        resource_ids currently in TRAIN mode (used for DP group membership).

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

        # Unified large RayWorkerGroup registry.  In the single-wg architecture
        # there is only one entry keyed as "elastic_wg".
        self._elastic_wg_registry: dict[str, RayWorkerGroup] = {}

        # elastic_unit_ranks: resource_id → list of wg-local ranks for that unit
        # e.g. {"elastic_0": [0,1], "elastic_1": [2,3], ...}  (TP×PP×CP ranks each)
        self._elastic_unit_ranks: dict[str, list[int]] = {}

        # Set of resource_ids currently in TRAIN mode (active in DP group)
        self._elastic_active_units: set[str] = set()

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

        # Initialize required_samples to be DP-size aware from the start.
        # Parent class sets required_samples = ppo_mini_batch_size * require_batches,
        # which may not be divisible by the initial DP size.
        self._update_required_samples()

        logger.info("[ElasticTrainer] Initialized with elastic DP support")

    # =========================================================================
    # Worker Initialization Overrides
    # =========================================================================

    def _is_fully_elastic(self) -> bool:
        """ElasticTrainer always operates in fully-elastic mode.

        All actor GPUs come from elastic_worker_groups registered via
        register_elastic_worker_group(), never from a static ResourcePool.
        trainer.nnodes / trainer.n_gpus_per_node describe the *total elastic
        GPU budget* (used to create elastic_worker_groups in main.py) and are
        NOT used to allocate a fixed trainer ResourcePool.
        """
        return True

    def _create_actor_rollout_classes(self):
        """
        ElasticTrainer has no fixed actor ResourcePool; all actor GPUs come
        from elastic_worker_groups wired in via register_elastic_worker_group().
        Skip the standard class-creation step entirely.
        """
        logger.info("[ElasticTrainer] Fully-elastic mode: skipping fixed actor rollout class creation")

    def _init_models(self):
        """
        Skip actor_wg initialization (no fixed actor resource pool).
        Critic / ref-policy / reward-model are still initialised as usual if present.
        actor_wg / actor_rollout_wg will be set later when the first elastic wg is registered.
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

        if resource_id not in self._elastic_active_units:
            logger.warning(f"[ElasticTrainer] {resource_id} not in active training units")
            return False

        unit_ranks = self._elastic_unit_ranks.get(resource_id, [])
        wg = self._get_elastic_wg()

        try:
            # Step 1: Remove from training DP group (rebuild DP without this rank)
            await self.remove_elastic_actor(resource_id)

            # Step 2: Offload actor weights to CPU on the workers for this unit.
            # GPU memory freed for rollout server.
            if wg is not None and unit_ranks:
                try:
                    futures = wg.execute_rank_method(
                        ranks=unit_ranks,
                        method_name="switch_to_rollout",
                        param_version=param_version,
                    )
                    results = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
                    if not all(results):
                        logger.warning(f"[ElasticTrainer] Some workers failed switch_to_rollout for {resource_id}")
                except AttributeError:
                    # Fallback: execute_all if execute_rank_method is unavailable
                    futures = wg.execute_all("switch_to_rollout", param_version=param_version)
                    results = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
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

        if resource_id not in self._elastic_unit_ranks:
            logger.warning(
                f"[ElasticTrainer] {resource_id} not in unit_ranks registry. "
                "Call register_elastic_unit_ranks() at init time."
            )
            return False

        unit_ranks = self._elastic_unit_ranks[resource_id]
        wg = self._get_elastic_wg()

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

            # Step 2: Compute the new train world ranks after adding this unit back.
            # Active units already excludes this resource_id (it's in rollout mode),
            # so the new set is: active_units ∪ {resource_id}.
            new_active = self._elastic_active_units | {resource_id}
            new_train_world_ranks: list[int] = []
            for rid in new_active:
                new_train_world_ranks.extend(self._elastic_unit_ranks.get(rid, []))
            new_train_world_ranks = sorted(set(new_train_world_ranks))

            # Load actor weights to GPU on the workers for this unit.
            if wg is not None and unit_ranks:
                try:
                    futures = wg.execute_rank_method(
                        ranks=unit_ranks,
                        method_name="switch_to_train",
                        new_train_world_ranks=new_train_world_ranks,
                        param_version=param_version,
                    )
                    results = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
                    if not all(results):
                        logger.warning(f"[ElasticTrainer] Some workers failed switch_to_train for {resource_id}")
                except AttributeError:
                    # Fallback: execute_all if execute_rank_method is unavailable
                    futures = wg.execute_all(
                        "switch_to_train",
                        new_train_world_ranks=new_train_world_ranks,
                        param_version=param_version,
                    )
                    results = await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
                except Exception as e:
                    logger.warning(f"[ElasticTrainer] switch_to_train on worker failed: {e}")

            # Step 3: Add to training DP group (rebuild DP with this rank)
            await self.add_elastic_actor(
                resource_id=resource_id,
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
        Register the unified large elastic RayWorkerGroup.

        In the single-wg architecture all elastic units share the same
        RayWorkerGroup; call this once with resource_id="elastic_wg".

        Also promotes the wg to actor_wg in fully-elastic mode so that the
        rest of the trainer (checkpoint manager, param-sync, etc.) has a
        valid handle.

        Args:
            resource_id: Convention key, use "elastic_wg" for the unified wg.
            worker_group: The single large RayWorkerGroup.
        """
        self._elastic_wg_registry[resource_id] = worker_group

        # In fully-elastic mode (no fixed actor pool), promote to actor_wg
        # so that the rest of the trainer has a valid handle.
        if self._is_fully_elastic() and getattr(self, "actor_wg", None) is None:
            self.actor_wg = worker_group
            self.actor_rollout_wg = worker_group
            logger.info(f"[ElasticTrainer] Fully-elastic mode: promoted '{resource_id}' as actor_wg")

        logger.info(f"[ElasticTrainer] Registered elastic worker group: {resource_id}")

    def register_elastic_unit_ranks(self, unit_ranks: dict) -> None:
        """
        Register the per-unit rank mapping.

        Args:
            unit_ranks: Mapping of resource_id → list of wg-local ranks.
                e.g. {"elastic_0": [0,1], "elastic_1": [2,3], ...}

        On registration, ALL units are assumed to be in TRAIN mode (initial
        state: all elastic GPUs are allocated to training).
        """
        self._elastic_unit_ranks = dict(unit_ranks)
        # Initially all units are in TRAIN mode
        self._elastic_active_units = set(unit_ranks.keys())

        # Initialize _elastic_dp_active_ranks on the worker group so that
        # the elastic dispatch strategy can route correctly from the start,
        # before the first rebuild_dp_group call.
        wg = self._get_elastic_wg()
        if wg is not None:
            all_ranks = sorted(set(r for ranks in unit_ranks.values() for r in ranks))
            wg._elastic_dp_active_ranks = all_ranks
            logger.info(f"[ElasticTrainer] Initialized _elastic_dp_active_ranks={all_ranks} on elastic_wg")

        logger.info(
            f"[ElasticTrainer] Registered {len(unit_ranks)} elastic unit rank mapping(s): , ".join(
                f"{rid}={ranks}" for rid, ranks in unit_ranks.items()
            )
        )

    def _get_elastic_wg(self) -> Optional[RayWorkerGroup]:
        """Return the single large elastic RayWorkerGroup (if registered)."""
        return self._elastic_wg_registry.get("elastic_wg")

    # =========================================================================
    # Elastic Actor Management (lower-level, used internally)
    # =========================================================================

    async def add_elastic_actor(
        self,
        resource_id: str,
        actor_worker_group: RayWorkerGroup = None,
        actor_handles: list = None,
        param_version: int = 0,
    ) -> bool:
        """
        Add an elastic unit back into the training DP group and trigger
        a rebuild via rebuild_dp_group().

        In the single-wg architecture, all units live in the same
        RayWorkerGroup; this method simply marks the unit as ACTIVE and
        schedules a DP rebuild that will include its ranks.

        Args:
            resource_id: Unique identifier for the elastic unit (e.g. "elastic_0").
            actor_worker_group: Unused in single-wg mode; kept for API compat.
            actor_handles: Unused; kept for API compatibility.
            param_version: Current parameter version.

        Returns:
            True if successfully queued, False otherwise.
        """
        async with self._dp_rebuild_lock:
            if resource_id in self._elastic_active_units:
                logger.warning(f"[ElasticTrainer] {resource_id} already in active training units")
                return False
            if resource_id not in self._elastic_unit_ranks:
                logger.warning(
                    f"[ElasticTrainer] {resource_id} not in unit_ranks registry; call register_elastic_unit_ranks first"
                )
                return False

            self._elastic_active_units.add(resource_id)
            self._elastic_actor_versions[resource_id] = param_version
            self._pending_elastic_adds.append(resource_id)
            self._dp_rebuild_pending = True
            self._total_elastic_adds += 1

            logger.info(
                f"[ElasticTrainer] Queued elastic unit {resource_id} for ADD "
                f"(ranks={self._elastic_unit_ranks[resource_id]}, "
                f"param_version={param_version}). DP rebuild deferred."
            )

        # Apply the pending DP change immediately if we're not mid-step
        if not self._is_training_step:
            await self._apply_pending_dp_changes()

        return True

    async def remove_elastic_actor(self, resource_id: str) -> bool:
        """
        Remove an elastic unit from the training DP group and trigger
        a rebuild via rebuild_dp_group().

        In the single-wg architecture, all units live in the same
        RayWorkerGroup; this method marks the unit as INACTIVE and
        schedules a DP rebuild that excludes its ranks.

        Called by switch_elastic_to_rollout() before the actor weights are
        offloaded.

        Args:
            resource_id: Unique identifier for the elastic unit (e.g. "elastic_0").

        Returns:
            True if successfully queued for removal, False otherwise.
        """
        async with self._dp_rebuild_lock:
            if resource_id not in self._elastic_active_units:
                logger.warning(f"[ElasticTrainer] {resource_id} not found in active training units")
                return False

            self._pending_elastic_removes.append(resource_id)
            self._dp_rebuild_pending = True
            self._total_elastic_removes += 1

            logger.info(
                f"[ElasticTrainer] Queued elastic unit {resource_id} for REMOVE "
                f"(ranks={self._elastic_unit_ranks.get(resource_id, [])}). DP rebuild deferred."
            )

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
                # Compute new active ranks from the unit-rank registry.
                # _elastic_active_units already reflects adds; removes are
                # still pending, so exclude them explicitly.
                remove_set = set(self._pending_elastic_removes)
                active_after = self._elastic_active_units - remove_set

                new_world_ranks: list[int] = []
                for rid in active_after:
                    new_world_ranks.extend(self._elastic_unit_ranks.get(rid, []))
                new_world_ranks = sorted(set(new_world_ranks))

                await self._coordinate_dp_rebuild(new_world_ranks=new_world_ranks)

                # Finalise removals: remove from active set
                for rid in self._pending_elastic_removes:
                    self._elastic_active_units.discard(rid)
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

    async def _coordinate_dp_rebuild(self, new_world_ranks: list[int]):
        """
        Call rebuild_dp_group(new_world_ranks) on the single large elastic wg.

        Because all elastic workers share the same PyTorch distributed world
        (dist.init_process_group was called once for the whole wg), a single
        broadcast to all workers is sufficient.  dist.new_group() is a global
        collective; every rank in the world must call it simultaneously, which
        is guaranteed here because execute_all() sends the call to every worker.

        Args:
            new_world_ranks: Sorted list of wg-local ranks that should form
                             the new DP group after the rebuild.
        """
        wg = self._get_elastic_wg()
        if wg is None:
            logger.warning("[ElasticTrainer] _coordinate_dp_rebuild: no elastic wg registered, skipping")
            return

        await self._trigger_rebuild_on_worker_group(wg, new_world_ranks)

    async def _trigger_rebuild_on_worker_group(
        self,
        worker_group: RayWorkerGroup,
        new_world_ranks: list[int],
    ):
        """Call rebuild_dp_group on every worker in the group.

        After the rebuild completes ``worker_group._elastic_dp_active_ranks`` is
        updated to ``new_world_ranks`` so that the elastic DP dispatch strategy
        (``make_elastic_dp_dispatch_fn``) correctly routes data only to the
        active ranks on the next dispatch call.

        The elastic dispatch strategy (used by ``compute_log_prob``,
        ``compute_ref_log_prob``, ``update_actor`` in ElasticActorWorker) reads
        ``worker_group._elastic_dp_active_ranks`` directly instead of relying on
        the ``_dispatch_info`` / ``_collect_info`` cache, so no manual cache
        patching is required here.
        """
        try:
            futures = worker_group.execute_all("rebuild_dp_group", new_world_ranks=new_world_ranks)
            await asyncio.get_event_loop().run_in_executor(None, lambda: ray.get(futures))
        except Exception as e:
            logger.error(f"[ElasticTrainer] rebuild_dp_group failed: {e}")
            raise

        # Update the elastic DP active ranks on the controller side.
        # The elastic dispatch strategy reads this attribute to know which
        # global ranks are currently participating in the DP group.
        self._update_elastic_dp_active_ranks(worker_group, new_world_ranks)

        logger.info(
            f"[ElasticTrainer] DP rebuild complete: "
            f"new_world_ranks={new_world_ranks}, "
            f"active_dp_size={len(new_world_ranks)}"
        )

    def _update_elastic_dp_active_ranks(
        self,
        worker_group: RayWorkerGroup,
        new_world_ranks: list[int],
    ) -> None:
        """Update ``worker_group._elastic_dp_active_ranks`` after a DP rebuild.

        This attribute is read by the elastic DP dispatch strategy
        (``dispatch_elastic_dp_dataproto`` in ``decorator.py``) to decide
        which global ranks receive data shards.  Inactive ranks receive
        ``None`` and their workers return ``None`` immediately, which is
        then filtered out by ``collect_elastic_dp_dataproto``.

        Args:
            worker_group: The RayWorkerGroup whose dispatch attribute to update.
            new_world_ranks: Sorted list of global ranks now active in the DP group.
        """
        worker_group._elastic_dp_active_ranks = list(new_world_ranks)
        logger.info(f"[ElasticTrainer] _elastic_dp_active_ranks updated: {new_world_ranks}")

    # =========================================================================
    # 7.4  Dynamic required_samples (DP-aware)
    # =========================================================================

    def _get_actor_cfg(self):
        """Shorthand accessor for actor_rollout_ref.actor config (may be None)."""
        return getattr(self.config.actor_rollout_ref, "actor", None)

    def _get_current_dp_size(self) -> int:
        """
        Return the number of DP replicas currently participating in training.

        In the single-wg architecture each elastic unit maps to exactly one
        DP replica (it covers the model-parallel ranks for that unit), so:
            DP size = number of currently active elastic units

        Falls back to actor_wg.world_size / gpus_per_unit before
        register_elastic_unit_ranks() is called.  gpus_per_unit depends on
        the actor strategy:
          - Megatron : TP × PP × CP  (from actor.megatron.*)
          - FSDP/FSDP2: ulysses_sequence_parallel_size  (from actor.fsdp_config.*)
        """
        if self._elastic_unit_ranks:
            # Each registered unit is one DP replica by definition.
            return max(len(self._elastic_active_units), 1)

        # Fallback: derive from the static actor_wg world size.
        gpus_per_unit = self._get_gpus_per_elastic_unit()
        if getattr(self, "actor_wg", None) is not None:
            return max(self.actor_wg.world_size // gpus_per_unit, 1)
        return 1

    def _get_gpus_per_elastic_unit(self) -> int:
        """
        Return the number of GPUs per elastic unit based on actor strategy.

        Megatron : TP × PP × CP
        FSDP/FSDP2 : ulysses_sequence_parallel_size (SP acts as the
                     intra-unit parallelism dimension; no TP/PP/CP)
        """
        actor_cfg = self._get_actor_cfg()
        if actor_cfg is None:
            return 1

        strategy = getattr(actor_cfg, "strategy", "fsdp")

        if strategy == "megatron":
            megatron_cfg = getattr(actor_cfg, "megatron", None) or actor_cfg
            tp = getattr(megatron_cfg, "tensor_model_parallel_size", 1)
            pp = getattr(megatron_cfg, "pipeline_model_parallel_size", 1)
            cp = getattr(megatron_cfg, "context_parallel_size", 1)
            return max(tp * pp * cp, 1)
        else:
            # FSDP / FSDP2: intra-unit parallelism = Ulysses SP size
            fsdp_cfg = getattr(actor_cfg, "fsdp_config", None) or actor_cfg
            sp = getattr(fsdp_cfg, "ulysses_sequence_parallel_size", 1)
            return max(sp, 1)

    def _get_dp_aligned_mini_batch_size(self) -> int:
        """
        Return a global ``mini_batch_size`` (in responses) that satisfies all
        three constraints required by ``train_mini_batch``:

        1. ``mini_batch_size % dp_size == 0``
           — so ``mini_batch_size_per_gpu = mini_batch_size // dp_size`` is an integer.

        2. ``(total_responses / dp_size) % (mini_batch_size / dp_size) == 0``
           ⟺ ``total_responses % mini_batch_size == 0``
           — so the per-rank batch divides evenly into sub-mini-batches.

        3. ``mini_batch_size % rollout_n == 0``
           — so ``prompts_per_mini_batch = mini_batch_size // rollout_n`` is exact,
             avoiding floor-division truncation that would break constraint (2).

        We align ``base * rollout_n`` up to the nearest multiple of
        ``LCM(dp_size, rollout_n)`` which is the tightest unit satisfying (1) and (3).

        Returns:
            int: Aligned global mini_batch_size (response count).
        """
        import math

        base = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        rollout_n = self.config.actor_rollout_ref.rollout.n
        dp_size = self._get_current_dp_size()
        # Align to LCM(dp_size, rollout_n) so the result is divisible by both.
        lcm = dp_size * rollout_n // math.gcd(dp_size, rollout_n)
        mini_batch_size = base * rollout_n
        if mini_batch_size % lcm != 0:
            mini_batch_size = math.ceil(mini_batch_size / lcm) * lcm
        return mini_batch_size

    def _update_required_samples(self) -> None:
        """
        Recompute ``required_samples`` based on the DP-aligned mini_batch_size.

        Terminology
        -----------
        * ``required_samples``  — number of **samples** (prompts) to pull from
          the queue.  Each sample in the queue already contains all ``rollout.n``
          responses for one prompt, so this is purely a **prompt count**.
        * ``aligned_mini_batch`` — total number of **responses** that will be
          handed to ``train_mini_batch``.  ``megatron_workers`` internally
          multiplies ``ppo_mini_batch_size × rollout_n`` and divides by
          ``dp_size``, so the assertion it checks is::

              (ppo_mini_batch_size × rollout_n) % dp_size == 0

        Formula::

            aligned_mini_batch = ceil(ppo_mini_batch_size × rollout_n / dp_size) × dp_size
            required_samples   = ceil(aligned_mini_batch / rollout_n / dp_size) × dp_size
                                 × require_batches
                                 (prompt count, NOT response count)

        The double alignment (once for responses, once for prompts) ensures:
        1. ``aligned_mini_batch % dp_size == 0``          — satisfies ``train_mini_batch`` assertion.
        2. ``required_samples × rollout_n % dp_size == 0`` — satisfies ``_balance_batch``
           (``get_seqlen_balanced_partitions`` requires total responses % dp_size == 0).

        Also propagates the new ``required_samples`` to the Rollouter so that
        its staleness / pause thresholds stay in sync with the Trainer's batch
        expectations.  The remote call is fire-and-forget (no ``await`` /
        ``ray.get``); the Rollouter updates atomically under its own lock.

        Called automatically at the end of ``_apply_pending_dp_changes()``.
        """
        import math

        require_batches = self.config.async_training.require_batches
        rollout_n = self.config.actor_rollout_ref.rollout.n
        dp_size = self._get_current_dp_size()

        aligned_mini_batch = self._get_dp_aligned_mini_batch_size()

        # aligned_mini_batch is the global mini-batch size (responses) that
        # train_mini_batch will use.  The per-rank batch handed to train_mini_batch
        # is  total_responses / dp_size, and train_mini_batch further splits it
        # into sub-mini-batches of size  aligned_mini_batch / dp_size.
        # For the inner assert to pass we need:
        #
        #   (total_responses / dp_size) % (aligned_mini_batch / dp_size) == 0
        #   ⟺  total_responses % aligned_mini_batch == 0
        #   ⟺  (required_samples × rollout_n) % aligned_mini_batch == 0
        #
        # So required_samples × rollout_n must be a multiple of aligned_mini_batch.
        # Equivalently, required_samples must be a multiple of
        #   aligned_mini_batch / rollout_n  (= prompts_per_mini_batch).
        #
        # We also need (required_samples × rollout_n) % dp_size == 0 for
        # _balance_batch (get_seqlen_balanced_partitions).  Both constraints
        # are satisfied by aligning to LCM(prompts_per_mini_batch, dp_size).

        # prompts_per_mini_batch: how many prompts fill one global mini-batch
        prompts_per_mini_batch = aligned_mini_batch // rollout_n

        # LCM of prompts_per_mini_batch and dp_size ensures both constraints.
        alignment_unit = prompts_per_mini_batch * dp_size // math.gcd(prompts_per_mini_batch, dp_size)

        # require_batches mini-batches worth of prompts, rounded up to alignment_unit
        base_prompts = prompts_per_mini_batch * require_batches
        aligned_prompts = math.ceil(base_prompts / alignment_unit) * alignment_unit
        new_required = aligned_prompts

        old_required = self.required_samples
        self.required_samples = new_required
        logger.info(
            f"[ElasticTrainer] required_samples updated: {old_required} → {new_required} "
            f"(dp_size={dp_size}, aligned_mini_batch={aligned_mini_batch}, "
            f"require_batches={require_batches})"
        )

        # Propagate to the Rollouter so its staleness/pause thresholds match.
        # Fire-and-forget: the remote call is non-blocking; the Rollouter will
        # atomically update max_required_samples and wake any paused coroutines.
        # Skip during __init__ when rollouter is not yet registered.
        if hasattr(self, "rollouter") and self.rollouter is not None:
            self.rollouter.update_required_samples.remote(new_required)
            logger.info(f"[ElasticTrainer] Notified Rollouter: update_required_samples({new_required})")

        # Update metrics_aggregator.total_gpus to reflect the current actual GPU
        # count so that perf/throughput is computed correctly.
        # In fully-elastic mode the static config gives 0 GPUs; we derive the real
        # count from the registered actor worker groups and the rollout config.
        self._update_metrics_aggregator_total_gpus()

    def _update_metrics_aggregator_total_gpus(self) -> None:
        """Sync metrics_aggregator.total_gpus with the current actual GPU count.

        In fully-elastic mode the static config gives trainer.n_gpus_per_node=0,
        so MetricsAggregator is initialised with total_gpus=0 and
        perf/throughput would cause a ZeroDivisionError.  This method recomputes
        the true GPU count via _get_n_gpus() and updates the aggregator.
        """
        if not hasattr(self, "metrics_aggregator"):
            return

        total_gpus = self._get_n_gpus()
        if total_gpus > 0 and self.metrics_aggregator.total_gpus != total_gpus:
            logger.info(
                f"[ElasticTrainer] metrics_aggregator.total_gpus updated: "
                f"{self.metrics_aggregator.total_gpus} → {total_gpus}"
            )
            self.metrics_aggregator.total_gpus = total_gpus

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
        1. _elastic_on_before_fit_step() — coordinator executes pending switch
           (Train GPU idle after previous step; switches happen here FIRST)
        2. _apply_pending_dp_changes() — DP rebuild + update required_samples
           (must complete before pulling data so batch size matches new DP)
        3. _fit_generate() — collect data sized for the NEW DP topology
        4. Training compute (GPU busy)
        5. _fit_update_weights() — parameter sync to rollout replicas

        Why switch BEFORE pulling data (not after):
        - After a switch the DP size changes, which changes required_samples.
        - If we pulled data first (old required_samples) then switched, the
          batch we hand to update_actor would be sized for the OLD DP.
        - By switching first we guarantee: batch_size == aligned_mini_batch
          for the current DP, so train_mini_batch's assertion never fires.
        """
        self._is_training_step = False
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}
        self._fit_start_profile()

        try:
            # Phase 1: elastic switch hook — Train GPU idle after previous step
            await self._elastic_on_before_fit_step()

            # Phase 2: apply any deferred DP changes and update required_samples
            # MUST happen before _fit_generate() so data is pulled at the
            # correct size for the new DP topology.
            if self._dp_rebuild_pending:
                try:
                    await self._apply_pending_dp_changes()
                except Exception as e:
                    logger.error(f"[ElasticTrainer] DP change application failed: {e}")
                    self._dp_rebuild_pending = False
                    self._pending_elastic_adds.clear()
                    self._pending_elastic_removes.clear()

            # Phase 3: collect data sized for the current (post-switch) DP
            batch = await self._fit_generate(None)

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

    def _update_actor(self, batch):
        """
        Override to dynamically align ``ppo_mini_batch_size`` with the current
        DP size before calling ``update_actor``.

        After an elastic DP rebuild the DP size may no longer divide the
        configured ``ppo_mini_batch_size`` evenly (e.g. mini_batch=768, dp=5).
        ``engine_workers.train_mini_batch`` asserts::

            mini_batch_size % dp_size == 0

        We round ``ppo_mini_batch_size`` UP to the nearest multiple of
        ``dp_size`` so that the assertion never fires regardless of the
        current (dynamic) DP topology.  The adjusted value only affects the
        per-rank mini-batch granularity; the total number of gradient-update
        steps per epoch is preserved to within ±1 step.

        Only active in the ``use_legacy_worker_impl == "disable"`` code path;
        the legacy path is left unchanged.
        """

        from verl.protocol import DataProto
        from verl.trainer.distillation.losses import is_distillation_enabled
        from verl.utils import tensordict_utils as tu
        from verl.utils.py_functional import rename_dict
        from verl.workers.utils.padding import left_right_2_no_padding

        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature

        if self.use_legacy_worker_impl != "disable":
            # Legacy path – unchanged.
            actor_output = self.actor_rollout_wg.update_actor(batch)
            return actor_output

        # ── new-style (tensordict) path ───────────────────────────────────
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )

        # Use DP-aligned mini_batch_size so that train_mini_batch's assertion
        # (mini_batch_size % dp_size == 0) always holds after elastic rebuilds.
        ppo_mini_batch_size = self._get_dp_aligned_mini_batch_size()
        logger.info(
            f"[ElasticTrainer] _update_actor using aligned mini_batch_size={ppo_mini_batch_size} "
            f"(dp_size={self._get_current_dp_size()})"
        )

        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle

        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        return actor_output

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
    # Metrics
    # =========================================================================

    def _get_n_gpus(self) -> int:
        """Return the current total GPU count across all active worker groups.

        In fully-elastic mode the static ResourcePoolManager reports 0 GPUs
        because trainer.n_gpus_per_node=0.  We derive the real count from the
        live worker group world sizes instead.
        """
        # Trainer GPUs: count only the ranks currently in active training units
        trainer_gpus = 0
        if self._elastic_unit_ranks and self._elastic_active_units:
            for rid in self._elastic_active_units:
                trainer_gpus += len(self._elastic_unit_ranks.get(rid, []))
        elif hasattr(self, "actor_wg") and self.actor_wg is not None:
            trainer_gpus = self.actor_wg.world_size

        # Rollout GPUs: static config (rollout resources are not elastic)
        rollout_gpus = getattr(self.config.rollout, "nnodes", 0) * getattr(self.config.rollout, "n_gpus_per_node", 0)

        static_n_gpus = self.resource_pool_manager.get_n_gpus()
        # Prefer the live count; fall back to static if we haven't registered
        # any elastic workers yet.
        return trainer_gpus + rollout_gpus if (trainer_gpus + rollout_gpus) > 0 else static_n_gpus

    def _fit_collect_metrics(self, batch):
        """Override to supply the dynamically-computed GPU count for throughput."""
        from verl.trainer.ppo.metric_utils import (
            compute_data_metrics,
            compute_throughout_metrics,
            compute_timing_metrics,
            compute_variance_proxy_metrics,
        )

        metrics = self.metrics
        timing_raw = self.timing_raw

        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self._get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

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
        """Total number of active training elastic units."""
        return len(self._elastic_active_units)

    def get_elastic_statistics(self) -> dict:
        """Elastic-specific statistics for monitoring."""
        return {
            "elastic_trainer/num_elastic_active_units": len(self._elastic_active_units),
            "elastic_trainer/num_elastic_units_total": len(self._elastic_unit_ranks),
            "elastic_trainer/total_adds": self._total_elastic_adds,
            "elastic_trainer/total_removes": self._total_elastic_removes,
            "elastic_trainer/total_dp_rebuilds": self._total_dp_rebuilds,
            "elastic_trainer/last_dp_rebuild_time": self._last_dp_rebuild_time,
            "elastic_trainer/consumption_rate_ema": self._consumption_rate_ema or 0.0,
            "elastic_trainer/dp_rebuild_pending": self._dp_rebuild_pending,
        }

    # =========================================================================
    # Override balance_batch — use elastic DP size
    # =========================================================================

    def _balance_batch(self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Override to use the *current* elastic DP size for sequence-length balancing.

        The parent implementation queries ``actor_rollout_wg._dispatch_info``,
        which is a **static cache** populated at worker-group creation time.
        After an elastic DP rebuild the cache still holds the old dp_size (e.g.
        6) while the actual DP group now has fewer ranks (e.g. 5).  Passing the
        stale value to ``get_seqlen_balanced_partitions`` causes:

            AssertionError: 12320 % 6 != 0

        We replace the stale lookup with ``_get_current_dp_size()``, which
        always reflects the live count of active elastic units.
        """
        import torch

        from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance

        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)

        # Always use the live elastic DP size — never the stale dispatch_info cache.
        dp_size = self._get_current_dp_size()

        if keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)

        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
