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

Monitors rollout production rate vs. train consumption rate and decides
when to switch elastic resources between rollout and train modes.

Switch execution model
----------------------
All role switches are **deferred to training boundaries** so the Train GPU
is guaranteed idle when the switch executes:

    monitoring loop  →  set _pending_action
    ElasticTrainer._elastic_on_before_fit_step()  →  on_before_fit_step()
                                                  →  _execute_action()

Train → Rollout switch sequence
---------------------------------
1. [ElasticTrainer]          remove_elastic_actor()   (DP rebuild without this rank)
2. [ElasticActorWorker]      switch_to_rollout()      (offload actor weights to CPU)
   (called via ElasticTrainer.switch_elastic_to_rollout)
3. [ElasticRollouter]        add_elastic_replica()    (wake_up rollout server + LB register)

Rollout → Train switch sequence
---------------------------------
1. [ElasticRollouter]        remove_elastic_replica() (sleep rollout server + abort in-flight)
2. [ElasticActorWorker]      switch_to_train()        (load weights to GPU)
   (called via ElasticTrainer.switch_elastic_to_train)
3. [ElasticTrainer]          add_elastic_actor()      (DP rebuild with this rank)

The strict ordering ensures GPU memory is always available at each step.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import ray

logger = logging.getLogger(__name__)


# ============================================================================
# Elastic Resource Info
# ============================================================================


@dataclass
class ElasticResourceInfo:
    """Tracks state of a single elastic resource."""

    resource_id: str
    current_mode: str  # "rollout" or "train"
    param_version: int = -1
    last_switch_time: float = 0.0
    is_healthy: bool = True

    # Only used by coordinator for grouping; actual worker ops go through
    # ElasticTrainer (train side) and ElasticRollouter (rollout side).
    worker_handles: list = field(default_factory=list)


# ============================================================================
# Elastic Coordinator
# ============================================================================


@ray.remote(num_cpus=1)
class ElasticCoordinator:
    """
    Central brain of elastic scheduling.

    Responsibilities
    ----------------
    1. Poll queue size, production rate (rollouter), consumption rate (trainer).
    2. Decide when to switch elastic resources (scale_rollout / scale_train).
    3. Defer execution to training boundaries so Train GPU is always idle.
    4. Delegate the actual switch to ElasticTrainer (which owns both sides):
         ElasticTrainer.switch_elastic_to_rollout(resource_id)
         ElasticTrainer.switch_elastic_to_train(resource_id)

    ElasticTrainer.switch_elastic_to_rollout / switch_elastic_to_train
    coordinate the full sequence (see module docstring) and call back into
    ElasticRollouter for the rollout-server side.
    """

    def __init__(
        self,
        elastic_rollouter,
        elastic_trainer,
        message_queue,
        elastic_resource_infos: list[dict],
        config: dict,
    ):
        """
        Args:
            elastic_rollouter: Ray handle to ElasticRollouter.
            elastic_trainer:   Ray handle to ElasticTrainer.
            message_queue:     Ray handle to MessageQueue (queue-size polling).
            elastic_resource_infos: List of dicts with keys:
                ``resource_id``   – unique string id
                ``initial_mode``  – "rollout" or "train"
                ``worker_handles``– (optional) list of Ray actor handles
            config: Dict with optional keys:
                high_watermark, low_watermark, cooldown_seconds, check_interval,
                ema_alpha, min_rollout_resources, min_train_resources,
                confidence_threshold, max_concurrent_switches.
        """
        self.rollouter = elastic_rollouter
        self.trainer = elastic_trainer
        self.message_queue = message_queue

        # Config
        self.high_watermark = config.get("high_watermark", 0.8)
        self.low_watermark = config.get("low_watermark", 0.3)
        self.cooldown_seconds = config.get("cooldown_seconds", 30.0)
        self.check_interval = config.get("check_interval", 2.0)
        self.ema_alpha = config.get("ema_alpha", 0.3)
        self.min_rollout_resources = config.get("min_rollout_resources", 0)
        self.min_train_resources = config.get("min_train_resources", 0)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.max_concurrent_switches = config.get("max_concurrent_switches", 1)

        # Resource registry
        self._resources: dict[str, ElasticResourceInfo] = {}
        for info in elastic_resource_infos:
            rid = info["resource_id"]
            self._resources[rid] = ElasticResourceInfo(
                resource_id=rid,
                current_mode=info.get("initial_mode", "rollout"),
                worker_handles=info.get("worker_handles", []),
            )

        # EMA rate tracking
        self._ema_production_rate: Optional[float] = None
        self._ema_consumption_rate: Optional[float] = None

        # Polling state
        self._last_queue_size: int = 0
        self._last_queue_capacity: int = 1
        self._last_produced_total: int = 0
        self._last_consumed_total: int = 0
        self._last_check_time: float = time.time()

        self._rate_history: deque = deque(maxlen=50)

        # Coordination
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._switch_lock = asyncio.Lock()

        # Pending action: set by monitoring loop, consumed by on_before_fit_step()
        self._pending_action: Optional[str] = None
        self._pending_action_confidence: float = 0.0

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
        logger.info("[ElasticCoordinator] Stopped")

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
            "pending_action": self._pending_action,
            "stats": dict(self._stats),
        }

    def _set_ema_rates_for_test(self, prod_rate, cons_rate):
        """Inject EMA rates directly – for unit testing only."""
        self._ema_production_rate = prod_rate
        self._ema_consumption_rate = cons_rate

    # =========================================================================
    # Training Boundary Hook
    # =========================================================================

    async def on_before_fit_step(self, step: int = 0) -> bool:
        """
        Called by ElasticTrainer after _fit_generate() returns and before
        training compute starts (Train GPU is idle).

        Consumes the pending action (if any) and delegates execution to
        ElasticTrainer, which owns the complete switch sequence.

        Returns:
            True if a switch was executed, False otherwise.
        """
        if self._pending_action is None:
            return False

        async with self._switch_lock:
            if self._pending_action is None:
                return False

            # Cooldown check
            if time.time() - self._stats.get("last_action_time", 0.0) < self.cooldown_seconds:
                return False

            action = self._pending_action
            self._pending_action = None
            self._pending_action_confidence = 0.0

        logger.info(f"[ElasticCoordinator] Executing '{action}' at training boundary (step={step})")
        await self._execute_action(action)
        return True

    # =========================================================================
    # Monitoring Loop
    # =========================================================================

    async def _monitoring_loop(self):
        while self._running:
            try:
                await self._run_single_coordination_step()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[ElasticCoordinator] Monitoring error: {e}")
            await asyncio.sleep(self.check_interval)

    async def _run_single_coordination_step(self):
        now = time.time()
        elapsed = now - self._last_check_time
        if elapsed <= 0:
            return

        queue_size, queue_capacity = await self._poll_queue_metrics()
        produced_total, consumed_total = await self._poll_rate_metrics()

        delta_produced = max(0, produced_total - self._last_produced_total)
        delta_consumed = max(0, consumed_total - self._last_consumed_total)

        if elapsed > 0:
            inst_prod = delta_produced / elapsed
            inst_cons = delta_consumed / elapsed
            if self._ema_production_rate is None:
                self._ema_production_rate = inst_prod
                self._ema_consumption_rate = inst_cons
            else:
                self._ema_production_rate = (
                    self.ema_alpha * inst_prod + (1 - self.ema_alpha) * self._ema_production_rate
                )
                self._ema_consumption_rate = (
                    self.ema_alpha * inst_cons + (1 - self.ema_alpha) * self._ema_consumption_rate
                )

        self._last_queue_size = queue_size
        self._last_queue_capacity = queue_capacity
        self._last_produced_total = produced_total
        self._last_consumed_total = consumed_total
        self._last_check_time = now

        queue_util = queue_size / max(queue_capacity, 1)
        self._rate_history.append(
            {
                "timestamp": now,
                "queue_util": queue_util,
                "production_rate": self._ema_production_rate or 0.0,
                "consumption_rate": self._ema_consumption_rate or 0.0,
            }
        )

        action, confidence = self._evaluate_action(queue_util)

        if action != "stable" and confidence >= self.confidence_threshold:
            if self._pending_action is None or confidence > self._pending_action_confidence:
                self._pending_action = action
                self._pending_action_confidence = confidence
                logger.info(
                    f"[ElasticCoordinator] Pending action queued: {action} "
                    f"(conf={confidence:.2f}), will execute at next training boundary"
                )

        self._stats["total_checks"] += 1

    async def _poll_queue_metrics(self) -> tuple[int, int]:
        try:
            queue_size = await asyncio.wrap_future(self.message_queue.get_queue_size.remote().future())
            queue_capacity = await asyncio.wrap_future(self.message_queue.get_max_queue_size.remote().future())
            return int(queue_size), int(queue_capacity)
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to poll queue metrics: {e}")
            return self._last_queue_size, self._last_queue_capacity

    async def _poll_rate_metrics(self) -> tuple[int, int]:
        produced_total = self._last_produced_total
        consumed_total = self._last_consumed_total
        try:
            produced_total = await asyncio.wrap_future(self.rollouter.get_total_produced_samples.remote().future())
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to poll production: {e}")
        try:
            consumed_total = await asyncio.wrap_future(self.trainer.get_total_consumed_samples.remote().future())
        except Exception as e:
            logger.warning(f"[ElasticCoordinator] Failed to poll consumption: {e}")
        return int(produced_total), int(consumed_total)

    def _evaluate_action(self, queue_util: float) -> tuple[str, float]:
        """
        Decide whether to scale rollout (more inference) or train (more training).

        Returns:
            (action, confidence) where action ∈ {"scale_rollout", "scale_train", "stable"}
        """
        prod_rate = self._ema_production_rate or 0.0
        cons_rate = self._ema_consumption_rate or 0.0

        n_rollout = sum(1 for r in self._resources.values() if r.current_mode == "rollout")
        n_train = sum(1 for r in self._resources.values() if r.current_mode == "train")

        can_add_rollout = n_train > self.min_train_resources
        can_add_train = n_rollout > self.min_rollout_resources

        # Queue too full → production bottleneck → need more rollout
        if queue_util > self.high_watermark and can_add_rollout:
            confidence = min(1.0, (queue_util - self.high_watermark) / (1.0 - self.high_watermark) + 0.6)
            return "scale_rollout", confidence

        # Queue too empty → consumption bottleneck → need more train
        if queue_util < self.low_watermark and can_add_train:
            if cons_rate > 0 and prod_rate > cons_rate * 1.2:
                confidence = min(1.0, (self.low_watermark - queue_util) / self.low_watermark + 0.6)
                return "scale_train", confidence

        # Rate-based fallback
        if prod_rate > cons_rate * 2.0 and queue_util > 0.5 and can_add_rollout:
            return "scale_rollout", 0.7
        if cons_rate > prod_rate * 2.0 and queue_util < 0.3 and can_add_train:
            return "scale_train", 0.7

        return "stable", 0.5

    # =========================================================================
    # Action Execution – delegates to ElasticTrainer
    # =========================================================================

    async def _execute_action(self, action: str):
        """
        Execute a scaling action by delegating to ElasticTrainer.

        ElasticTrainer owns the complete switch sequence for both sides
        (see ElasticTrainer.switch_elastic_to_rollout / switch_elastic_to_train).
        """
        if action == "scale_rollout":
            await self._switch_train_to_rollout()
        elif action == "scale_train":
            await self._switch_rollout_to_train()

    async def _switch_train_to_rollout(self, n: int = 1):
        """
        Switch N elastic resources: Train → Rollout.

        Delegates the full sequence to ElasticTrainer.switch_elastic_to_rollout()
        which handles:
          1. remove_elastic_actor (DP rebuild)
          2. worker.switch_to_rollout (offload actor to CPU)
          3. rollouter.add_elastic_replica (wake up rollout server)
        """
        candidates = [
            r
            for r in self._resources.values()
            if r.current_mode == "train" and r.is_healthy and (time.time() - r.last_switch_time) > self.cooldown_seconds
        ]
        if not candidates:
            logger.debug("[ElasticCoordinator] No candidates for train→rollout switch")
            return

        for resource_info in candidates[:n]:
            try:
                success = await asyncio.wrap_future(
                    self.trainer.switch_elastic_to_rollout.remote(
                        resource_id=resource_info.resource_id,
                        param_version=resource_info.param_version,
                    ).future()
                )
                if success:
                    resource_info.current_mode = "rollout"
                    resource_info.last_switch_time = time.time()
                    self._stats["rollout_scale_ups"] += 1
                    self._stats["total_switches"] += 1
                    logger.info(f"[ElasticCoordinator] {resource_info.resource_id} → rollout mode")
                else:
                    logger.warning(
                        f"[ElasticCoordinator] switch_elastic_to_rollout returned False for {resource_info.resource_id}"
                    )
            except Exception as e:
                logger.error(f"[ElasticCoordinator] Failed to switch {resource_info.resource_id} to rollout: {e}")
                resource_info.is_healthy = False
                self._stats["failed_switches"] += 1

        self._stats["last_action"] = "scale_rollout"
        self._stats["last_action_time"] = time.time()

    async def _switch_rollout_to_train(self, n: int = 1):
        """
        Switch N elastic resources: Rollout → Train.

        Delegates the full sequence to ElasticTrainer.switch_elastic_to_train()
        which handles:
          1. rollouter.remove_elastic_replica (sleep rollout server + abort)
          2. worker.switch_to_train (load actor to GPU)
          3. add_elastic_actor (DP rebuild)
        """
        candidates = [
            r
            for r in self._resources.values()
            if r.current_mode == "rollout"
            and r.is_healthy
            and (time.time() - r.last_switch_time) > self.cooldown_seconds
        ]
        if not candidates:
            logger.debug("[ElasticCoordinator] No candidates for rollout→train switch")
            return

        for resource_info in candidates[:n]:
            try:
                success = await asyncio.wrap_future(
                    self.trainer.switch_elastic_to_train.remote(
                        resource_id=resource_info.resource_id,
                        param_version=resource_info.param_version,
                    ).future()
                )
                if success:
                    resource_info.current_mode = "train"
                    resource_info.last_switch_time = time.time()
                    self._stats["train_scale_ups"] += 1
                    self._stats["total_switches"] += 1
                    logger.info(f"[ElasticCoordinator] {resource_info.resource_id} → train mode")
                else:
                    logger.warning(
                        f"[ElasticCoordinator] switch_elastic_to_train returned False for {resource_info.resource_id}"
                    )
            except Exception as e:
                logger.error(f"[ElasticCoordinator] Failed to switch {resource_info.resource_id} to train: {e}")
                resource_info.is_healthy = False
                self._stats["failed_switches"] += 1

        self._stats["last_action"] = "scale_train"
        self._stats["last_action_time"] = time.time()
