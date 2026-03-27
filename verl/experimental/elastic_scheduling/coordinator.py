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
Elastic Coordinator for VERL

This module implements the ElasticCoordinator which:
1. Monitors rollout production rate (from MessageQueue)
2. Monitors train consumption rate (from ElasticTrainer)
3. Decides when to switch elastic resources between rollout/train modes
4. Triggers role switches BEFORE parameter sync for consistency
5. Coordinates with ElasticRollouter and ElasticTrainer via Ray remote calls

Decision logic:
- If queue is filling up fast (production > consumption): elastic resources help ROLLOUT
- If queue is draining fast (consumption > production): elastic resources help TRAIN
- Role switch happens just before the next parameter sync cycle

"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import ray

logger = logging.getLogger(__name__)


# ============================================================================
# Elastic Coordinator
# ============================================================================


@dataclass
class ElasticResourceInfo:
    """Tracks state of a single elastic resource."""

    resource_id: str
    current_mode: str  # "rollout" or "train"
    param_version: int = -1
    last_switch_time: float = 0.0
    is_healthy: bool = True

    # Ray handles
    worker_handles: list = None  # Worker Ray actor handles
    rollout_replica: object = None  # RolloutReplica object (when in rollout mode)
    actor_wg: object = None  # RayWorkerGroup (when in train mode)


@ray.remote(num_cpus=1)
class ElasticCoordinator:
    """
    Elastic Coordinator for VERL

    Acts as the central brain of elastic scheduling. It:
    1. Continuously polls production rate (from MessageQueue) and consumption rate
       (from ElasticTrainer) to detect imbalance
    2. Decides when and which elastic resources to switch modes
    3. Executes the role switch BEFORE the next parameter sync, ensuring
       newly-joined rollout replicas get the latest parameters
    4. Coordinates DP group rebuild across all training workers

    Architecture:
        ElasticCoordinator (Ray actor)
            ├── Monitors queue size from MessageQueue
            ├── Polls ElasticRollouter for production stats
            ├── Polls ElasticTrainer for consumption stats
            └── Triggers switch via:
                ├── ElasticRollouter.add_elastic_replica() / remove_elastic_replica()
                └── ElasticTrainer.add_elastic_actor() / remove_elastic_actor()

    Switch timing:
        The coordinator hooks into the ElasticRollouter's parameter sync event.
        When a sync is about to be triggered (every trigger_parameter_sync_step steps),
        the coordinator first executes any pending role switches, then allows sync to proceed.
        This ensures new rollout replicas are in place before receiving fresh parameters.

    EMA-based rate monitoring:
        Production rate (samples/sec from rollout) and consumption rate
        (samples/sec consumed by trainer) are tracked with EMA smoothing.
        Queue utilization (queue_size / queue_capacity) serves as the primary signal.
    """

    def __init__(
        self,
        elastic_rollouter,  # Ray handle to ElasticRollouter
        elastic_trainer,  # Ray handle to ElasticTrainer
        message_queue,  # Ray handle to MessageQueue
        elastic_resource_infos: list[dict],  # list of {resource_id, worker_handles}
        config: dict,  # Coordinator config dict
    ):
        """
        Args:
            elastic_rollouter: Ray actor handle to ElasticRollouter
            elastic_trainer: Ray actor handle to ElasticTrainer
            message_queue: Ray actor handle to MessageQueue (for queue size polling)
            elastic_resource_infos: List of dicts with {resource_id, worker_handles}
            config: Configuration dict with keys:
                - high_watermark: float (default 0.8) - trigger rollout scale-up
                - low_watermark: float (default 0.3) - trigger train scale-up
                - cooldown_seconds: float (default 30.0) - min time between switches
                - check_interval: float (default 2.0) - polling interval in seconds
                - ema_alpha: float (default 0.3) - EMA smoothing factor
                - min_rollout_resources: int (default 0) - always in rollout mode
                - min_train_resources: int (default 0) - always in train mode
                - confidence_threshold: float (default 0.6) - min confidence for action
        """
        self.rollouter = elastic_rollouter
        self.trainer = elastic_trainer
        self.message_queue = message_queue

        # Parse config with defaults
        self.high_watermark = config.get("high_watermark", 0.8)
        self.low_watermark = config.get("low_watermark", 0.3)
        self.cooldown_seconds = config.get("cooldown_seconds", 30.0)
        self.check_interval = config.get("check_interval", 2.0)
        self.ema_alpha = config.get("ema_alpha", 0.3)
        self.min_rollout_resources = config.get("min_rollout_resources", 0)
        self.min_train_resources = config.get("min_train_resources", 0)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.max_concurrent_switches = config.get("max_concurrent_switches", 1)

        # Elastic resource registry
        self._resources: dict[str, ElasticResourceInfo] = {}
        for info in elastic_resource_infos:
            rid = info["resource_id"]
            self._resources[rid] = ElasticResourceInfo(
                resource_id=rid,
                current_mode=info.get("initial_mode", "rollout"),
                worker_handles=info.get("worker_handles", []),
            )

        # Rate monitoring state (EMA)
        self._ema_production_rate: Optional[float] = None
        self._ema_consumption_rate: Optional[float] = None

        # Queue state tracking
        self._last_queue_size: int = 0
        self._last_queue_capacity: int = 1
        self._last_produced_total: int = 0
        self._last_consumed_total: int = 0
        self._last_check_time: float = time.time()

        # Rate history for trend detection
        self._rate_history: deque = deque(maxlen=50)

        # Coordination state
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._switch_lock = asyncio.Lock()

        # Pending action flag: set by monitoring loop, consumed by on_before_fit_step()
        # This decouples decision (async monitor) from execution (training boundary)
        self._pending_action: Optional[str] = None
        self._pending_action_confidence: float = 0.0

        # Statistics
        self._stats = {
            "total_checks": 0,
            "rollout_scale_ups": 0,
            "train_scale_ups": 0,
            "total_switches": 0,
            "failed_switches": 0,
            "last_action": "none",
            "last_action_time": 0.0,
        }

        logger.info(
            f"[ElasticCoordinator] Initialized with {len(self._resources)} elastic resources. "
            f"high_wm={self.high_watermark}, low_wm={self.low_watermark}, "
            f"cooldown={self.cooldown_seconds}s"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    async def start(self):
        """Start the coordinator monitoring loop."""
        if self._running:
            logger.warning("[ElasticCoordinator] Already running")
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._monitoring_loop())
        logger.info("[ElasticCoordinator] Monitoring loop started")

    async def stop(self):
        """Stop the coordinator monitoring loop."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("[ElasticCoordinator] Monitoring loop stopped")

    def get_status(self) -> dict:
        """Get coordinator status for monitoring."""
        resources_by_mode = {
            "rollout": [r.resource_id for r in self._resources.values() if r.current_mode == "rollout"],
            "train": [r.resource_id for r in self._resources.values() if r.current_mode == "train"],
        }

        return {
            "running": self._running,
            "total_elastic_resources": len(self._resources),
            "resources_by_mode": resources_by_mode,
            "ema_production_rate": self._ema_production_rate,
            "ema_consumption_rate": self._ema_consumption_rate,
            "last_queue_utilization": self._last_queue_size / max(self._last_queue_capacity, 1),
            "stats": dict(self._stats),
        }

    # =========================================================================
    # Monitoring Loop
    # =========================================================================

    async def _monitoring_loop(self):
        """
        Main monitoring and coordination loop.

        Runs continuously while self._running is True.
        Each iteration:
        1. Polls queue size, production rate, consumption rate
        2. Updates EMA estimates
        3. Evaluates whether to switch elastic resources
        4. Executes switch if needed
        """
        logger.info("[ElasticCoordinator] Monitoring loop started")

        while self._running:
            try:
                await self._run_single_coordination_step()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[ElasticCoordinator] Error in monitoring loop: {e}")

            await asyncio.sleep(self.check_interval)

        logger.info("[ElasticCoordinator] Monitoring loop ended")

    async def _run_single_coordination_step(self):
        """Execute one round of monitoring and coordination."""
        now = time.time()
        elapsed = now - self._last_check_time
        if elapsed <= 0:
            return

        # Step 1: Poll queue metrics
        queue_size, queue_capacity = await self._poll_queue_metrics()

        # Step 2: Poll production/consumption totals
        produced_total, consumed_total = await self._poll_rate_metrics()

        # Step 3: Calculate delta since last check
        delta_produced = max(0, produced_total - self._last_produced_total)
        delta_consumed = max(0, consumed_total - self._last_consumed_total)

        # Step 4: Update EMA rates
        if elapsed > 0:
            instant_prod = delta_produced / elapsed
            instant_cons = delta_consumed / elapsed

            if self._ema_production_rate is None:
                self._ema_production_rate = instant_prod
                self._ema_consumption_rate = instant_cons
            else:
                self._ema_production_rate = (
                    self.ema_alpha * instant_prod + (1 - self.ema_alpha) * self._ema_production_rate
                )
                self._ema_consumption_rate = (
                    self.ema_alpha * instant_cons + (1 - self.ema_alpha) * self._ema_consumption_rate
                )

        # Update state
        self._last_queue_size = queue_size
        self._last_queue_capacity = queue_capacity
        self._last_produced_total = produced_total
        self._last_consumed_total = consumed_total
        self._last_check_time = now

        # Record rate history
        queue_util = queue_size / max(queue_capacity, 1)
        self._rate_history.append(
            {
                "timestamp": now,
                "queue_util": queue_util,
                "production_rate": self._ema_production_rate or 0.0,
                "consumption_rate": self._ema_consumption_rate or 0.0,
            }
        )

        # Step 5: Evaluate action
        action, confidence = self._evaluate_action(queue_util)

        # Step 6: Record pending action — actual execution happens at training boundary
        # (on_before_fit_step), so Train GPU is guaranteed idle when switch occurs.
        if action != "stable" and confidence >= self.confidence_threshold:
            if self._pending_action is None or confidence > self._pending_action_confidence:
                self._pending_action = action
                self._pending_action_confidence = confidence
                logger.info(
                    f"[ElasticCoordinator] Pending action queued: {action} "
                    f"(conf={confidence:.2f}), will execute at next training boundary"
                )

        self._stats["total_checks"] += 1

        logger.debug(
            f"[ElasticCoordinator] Check: queue={queue_util:.2f}, "
            f"prod={self._ema_production_rate:.2f}, cons={self._ema_consumption_rate:.2f}, "
            f"action={action} (conf={confidence:.2f})"
        )

    async def _poll_queue_metrics(self) -> tuple:
        """Poll current queue size and capacity from MessageQueue."""
        try:
            queue_size = await asyncio.wrap_future(self.message_queue.get_queue_size.remote().future())
            queue_capacity = await asyncio.wrap_future(self.message_queue.get_max_queue_size.remote().future())
            return int(queue_size), int(queue_capacity)
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to poll queue metrics: {e}")
            return self._last_queue_size, self._last_queue_capacity

    async def _poll_rate_metrics(self) -> tuple:
        """Poll production and consumption totals from rollouter and trainer."""
        produced_total = self._last_produced_total
        consumed_total = self._last_consumed_total

        # Poll production total from rollouter
        try:
            produced_total = await asyncio.wrap_future(self.rollouter.get_total_produced_samples.remote().future())
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to poll production: {e}")

        # Poll consumption total from trainer
        try:
            consumed_total = await asyncio.wrap_future(self.trainer.get_total_consumed_samples.remote().future())
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to poll consumption: {e}")

        return int(produced_total), int(consumed_total)

    def _evaluate_action(self, queue_util: float) -> tuple:
        """
        Evaluate what action to take based on current metrics.

        Returns:
            (action, confidence) where action is "scale_rollout", "scale_train", or "stable"
        """
        prod_rate = self._ema_production_rate or 0.0
        cons_rate = self._ema_consumption_rate or 0.0

        # Count resources in each mode
        n_rollout = sum(1 for r in self._resources.values() if r.current_mode == "rollout")
        n_train = sum(1 for r in self._resources.values() if r.current_mode == "train")
        # Check if we can switch any resource
        can_add_rollout = n_train > self.min_train_resources
        can_add_train = n_rollout > self.min_rollout_resources

        # High queue: production bottleneck → need more rollout resources
        if queue_util > self.high_watermark and can_add_rollout:
            # Production is the bottleneck
            confidence = min(1.0, (queue_util - self.high_watermark) / (1.0 - self.high_watermark) + 0.6)
            return "scale_rollout", confidence

        # Low queue: consumption is slow → need more train resources
        if queue_util < self.low_watermark and can_add_train:
            # Training is the bottleneck
            if cons_rate > 0 and prod_rate > cons_rate * 1.2:
                # Production clearly exceeds consumption
                confidence = min(1.0, (self.low_watermark - queue_util) / self.low_watermark + 0.6)
                return "scale_train", confidence

        # If production rate is much higher than consumption (and we have resources in train)
        if prod_rate > cons_rate * 2.0 and queue_util > 0.5 and can_add_rollout:
            confidence = 0.7
            return "scale_rollout", confidence

        # If consumption rate is much higher than production (and we have resources in rollout)
        if cons_rate > prod_rate * 2.0 and queue_util < 0.3 and can_add_train:
            confidence = 0.7
            return "scale_train", confidence

        return "stable", 0.5

    # =========================================================================
    # Training Boundary Hook (called by ElasticTrainer before each train step)
    # =========================================================================

    async def on_before_fit_step(self, step: int = 0) -> bool:
        """
        Called by ElasticTrainer after _fit_generate() returns (data collected)
        and before _fit_compute_reward() starts (GPU is idle).

        This is the preferred execution point for elastic role switches because:
        1. Train GPU is completely idle (waiting on queue just finished)
        2. No gradient updates are in progress
        3. DP rebuild cost is hidden within the data-wait period

        Returns:
            True if a role switch was executed, False otherwise.
        """
        action = self._pending_action
        if action is None:
            return False

        async with self._switch_lock:
            # Re-check after acquiring lock (another coroutine may have consumed it)
            if self._pending_action is None:
                return False

            # Check cooldown
            last_action_time = self._stats.get("last_action_time", 0.0)
            if time.time() - last_action_time < self.cooldown_seconds:
                logger.debug(
                    f"[ElasticCoordinator] on_before_fit_step: action suppressed by cooldown "
                    f"({time.time() - last_action_time:.1f}s < {self.cooldown_seconds}s)"
                )
                return False

            consumed_action = self._pending_action
            self._pending_action = None
            self._pending_action_confidence = 0.0

        logger.info(
            f"[ElasticCoordinator] Executing pending action '{consumed_action}' "
            f"at training boundary (step={step}), Train GPU is idle"
        )
        await self._execute_action(consumed_action, 1.0)
        return True

    # =========================================================================
    # Action Execution
    # =========================================================================

    async def _execute_action(self, action: str, confidence: float):
        """Execute a scaling action (must be called within _switch_lock or at training boundary)."""
        # Check cooldown (skip if called from on_before_fit_step which already checks)
        last_action_time = self._stats.get("last_action_time", 0.0)
        if time.time() - last_action_time < self.cooldown_seconds:
            logger.debug(
                f"[ElasticCoordinator] Action {action} suppressed by cooldown "
                f"({time.time() - last_action_time:.1f}s < {self.cooldown_seconds}s)"
            )
            return

        if action == "scale_rollout":
            await self._switch_train_to_rollout()
        elif action == "scale_train":
            await self._switch_rollout_to_train()

    async def _switch_train_to_rollout(self, n: int = 1):
        """
        Switch N elastic resources from Train mode to Rollout mode.

        Flow:
        1. Select N resources currently in train mode
        2. For each resource:
           a. Remove from ElasticTrainer (triggers DP rebuild on train workers)
           b. Wake up rollout engine on this resource
           c. Add to ElasticRollouter (ElasticParamSync will sync params next cycle)
        3. Update resource tracking
        """
        # Select candidates (in train mode, healthy, past cooldown)
        candidates = [
            r
            for r in self._resources.values()
            if r.current_mode == "train" and r.is_healthy and (time.time() - r.last_switch_time) > self.cooldown_seconds
        ]

        if not candidates:
            logger.debug("[ElasticCoordinator] No candidates for train->rollout switch")
            return

        to_switch = candidates[:n]
        logger.info(f"[ElasticCoordinator] Switching {len(to_switch)} resources: train -> rollout")

        for resource_info in to_switch:
            try:
                await self._do_switch_to_rollout(resource_info)
                self._stats["rollout_scale_ups"] += 1
                self._stats["total_switches"] += 1
            except Exception as e:
                logger.error(f"[ElasticCoordinator] Failed to switch {resource_info.resource_id} to rollout: {e}")
                resource_info.is_healthy = False
                self._stats["failed_switches"] += 1

        self._stats["last_action"] = "scale_rollout"
        self._stats["last_action_time"] = time.time()

    async def _switch_rollout_to_train(self, n: int = 1):
        """
        Switch N elastic resources from Rollout mode to Train mode.

        Flow:
        1. Select N resources currently in rollout mode
        2. For each resource:
           a. Remove from ElasticRollouter (stops new requests to this replica)
           b. Sleep rollout engine (releases GPU memory)
           c. Add to ElasticTrainer (triggers DP rebuild on train workers)
        3. Update resource tracking
        """
        # Select candidates (in rollout mode, healthy, past cooldown)
        candidates = [
            r
            for r in self._resources.values()
            if r.current_mode == "rollout"
            and r.is_healthy
            and (time.time() - r.last_switch_time) > self.cooldown_seconds
        ]

        if not candidates:
            logger.debug("[ElasticCoordinator] No candidates for rollout->train switch")
            return

        to_switch = candidates[:n]
        logger.info(f"[ElasticCoordinator] Switching {len(to_switch)} resources: rollout -> train")

        for resource_info in to_switch:
            try:
                await self._do_switch_to_train(resource_info)
                self._stats["train_scale_ups"] += 1
                self._stats["total_switches"] += 1
            except Exception as e:
                logger.error(f"[ElasticCoordinator] Failed to switch {resource_info.resource_id} to train: {e}")
                resource_info.is_healthy = False
                self._stats["failed_switches"] += 1

        self._stats["last_action"] = "scale_train"
        self._stats["last_action_time"] = time.time()

    async def _do_switch_to_rollout(self, resource_info: ElasticResourceInfo):
        """
        Perform the actual switch for one resource: Train -> Rollout.

        Steps:
        1. Remove from ElasticTrainer (which rebuilds train DP group without this resource)
        2. On the worker: sleep actor, wake up rollout engine
        3. Add to ElasticRollouter (as a new elastic rollout replica)
           - ElasticParamSync will send latest params at next sync cycle
        """
        resource_id = resource_info.resource_id
        logger.info(f"[ElasticCoordinator] Switching {resource_id}: train -> rollout")

        # Step 1: Remove from ElasticTrainer
        try:
            success = await asyncio.wrap_future(
                self.trainer.remove_elastic_actor.remote(resource_id=resource_id).future()
            )
            if not success:
                logger.warning(f"[ElasticCoordinator] Trainer remove failed for {resource_id}")
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to remove {resource_id} from trainer: {e}")

        # Step 2: Switch worker to rollout mode
        # The worker handles the sleep/wake_up internally via ElasticWorkerMixin
        if resource_info.worker_handles:
            switch_futures = []
            for handle in resource_info.worker_handles:
                try:
                    f = handle.switch_to_rollout.remote(
                        new_rollout_world_ranks=[],  # Will be determined by rollout replica
                        param_version=resource_info.param_version,
                    )
                    switch_futures.append(f.future())
                except Exception as e:
                    logger.warning(f"[ElasticCoordinator] Worker switch call failed: {e}")

            if switch_futures:
                results = await asyncio.gather(
                    *[asyncio.wrap_future(f) for f in switch_futures],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"[ElasticCoordinator] Worker switch result error: {result}")

        # Step 3: Activate the pre-registered elastic replica in the rollouter.
        # The replica was pre-registered via ElasticAgentLoopManager.create() at
        # initialisation time; we only need to pass resource_id + param_version.
        try:
            success = await asyncio.wrap_future(
                self.rollouter.add_elastic_replica.remote(
                    resource_id=resource_id,
                    param_version=resource_info.param_version,
                ).future()
            )
            if not success:
                logger.warning(f"[ElasticCoordinator] Rollouter add_elastic_replica failed for {resource_id}")

        except Exception as e:
            logger.error(f"[ElasticCoordinator] Failed to activate {resource_id} in rollouter: {e}")
            raise

        # Update state
        resource_info.current_mode = "rollout"
        resource_info.last_switch_time = time.time()
        logger.info(f"[ElasticCoordinator] {resource_id} switched to rollout mode")

    async def _do_switch_to_train(self, resource_info: ElasticResourceInfo):
        """
        Perform the actual switch for one resource: Rollout -> Train.

        Steps:
        1. Remove from ElasticRollouter (graceful drain of in-flight requests)
        2. On the worker: sleep rollout engine, load actor params from CPU
        3. Add to ElasticTrainer (which triggers DP rebuild to include this resource)
        4. Wait for parameter sync before allowing training to start
        """
        resource_id = resource_info.resource_id
        logger.info(f"[ElasticCoordinator] Switching {resource_id}: rollout -> train")

        # Step 1: Remove from ElasticRollouter (graceful drain)
        try:
            success = await asyncio.wrap_future(
                self.rollouter.remove_elastic_replica.remote(
                    resource_id=resource_id,
                ).future()
            )
            if not success:
                logger.warning(f"[ElasticCoordinator] Rollouter remove failed for {resource_id}")
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to remove {resource_id} from rollouter: {e}")

        # Step 2: Switch worker to train mode
        if resource_info.worker_handles:
            switch_futures = []
            for handle in resource_info.worker_handles:
                try:
                    f = handle.switch_to_train.remote(
                        new_train_world_ranks=[],  # Will be determined by trainer
                        param_version=resource_info.param_version,
                    )
                    switch_futures.append(f.future())
                except Exception as e:
                    logger.warning(f"[ElasticCoordinator] Worker switch call failed: {e}")

            if switch_futures:
                results = await asyncio.gather(
                    *[asyncio.wrap_future(f) for f in switch_futures],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"[ElasticCoordinator] Worker switch result error: {result}")

        # Step 3: Create actor worker group and add to ElasticTrainer
        try:
            actor_wg = await self._create_actor_wg_from_handles(resource_id, resource_info.worker_handles)
            if actor_wg is not None:
                success = await asyncio.wrap_future(
                    self.trainer.add_elastic_actor.remote(
                        resource_id=resource_id,
                        actor_worker_group=actor_wg,
                        actor_handles=resource_info.worker_handles or [],
                        param_version=resource_info.param_version,
                    ).future()
                )
                if success:
                    resource_info.actor_wg = actor_wg

        except Exception as e:
            logger.error(f"[ElasticCoordinator] Failed to add {resource_id} to trainer: {e}")
            raise

        # Clear rollout replica reference
        resource_info.rollout_replica = None

        # Update state
        resource_info.current_mode = "train"
        resource_info.last_switch_time = time.time()
        logger.info(f"[ElasticCoordinator] {resource_id} switched to train mode")

    async def _create_actor_wg_from_handles(self, resource_id: str, handles: list):
        """
        Create a RayWorkerGroup-compatible object from worker handles.

        Used when adding an elastic resource to ElasticTrainer.
        Creates a minimal worker group wrapper from existing Ray actor handles.
        """
        if not handles:
            return None

        try:
            from verl.single_controller.ray import RayWorkerGroup

            # Create a worker group wrapping the existing handles
            # Pass worker_handles directly so RayWorkerGroup wraps them
            # without trying to create new Ray actors.
            actor_wg = RayWorkerGroup(
                resource_pool=None,  # No new resources needed (reusing existing)
                ray_cls_with_init=None,  # Already initialized workers
                worker_handles=handles,  # Existing Ray actor handles
            )
            return actor_wg

        except TypeError:
            # Older version of RayWorkerGroup may not accept worker_handles
            # In that case, create a simple wrapper
            logger.warning(
                f"[ElasticCoordinator] RayWorkerGroup does not support worker_handles, "
                f"using minimal wrapper for {resource_id}"
            )

            class MinimalWorkerGroup:
                """Minimal wrapper providing execute_all() for direct handles."""

                def __init__(self, worker_handles):
                    self._handles = worker_handles
                    self.world_size = len(worker_handles)

                def execute_all(self, method_name: str, **kwargs):
                    """Call method on all workers, return futures."""
                    futures = []
                    for handle in self._handles:
                        method = getattr(handle, method_name)
                        futures.append(method.remote(**kwargs))
                    return futures

            return MinimalWorkerGroup(handles)

        except Exception as e:
            logger.error(f"[ElasticCoordinator] Failed to create actor worker group: {e}")
            return None
