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
Resource Coordination Module for VERL

Monitors congestion between rollout and training, and coordinates
dynamic resource allocation for optimal throughput.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

# Import for type hints
from .resource_manager import HybridEngineMode

if TYPE_CHECKING:
    from .resource_manager import ElasticResourceManager, HybridEngineResource

logger = logging.getLogger(__name__)


class SuggestedAction(Enum):
    """Suggested resource allocation action"""

    SCALE_UP_ROLLOUT = "scale_up_rollout"
    SCALE_UP_TRAIN = "scale_up_train"
    STABLE = "stable"
    REDUCE_ROLLOUT = "reduce_rollout"
    REDUCE_TRAIN = "reduce_train"


@dataclass
class CongestionMetrics:
    """Congestion metrics"""

    timestamp: float

    # Queue status
    queue_size: int
    queue_capacity: int
    queue_utilization: float  # 0-1

    # Production/consumption rates
    production_rate: float  # samples/sec (rollout produces)
    consumption_rate: float  # samples/sec (train consumes)

    # Congestion status
    is_rollout_congested: bool
    is_train_congested: bool
    imbalance_ratio: float  # production / consumption

    # Suggestions
    suggested_action: SuggestedAction
    confidence: float  # 0-1
    urgency: float  # 0-1, how urgent the action is

    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "timestamp": self.timestamp,
            "queue_utilization": self.queue_utilization,
            "production_rate": self.production_rate,
            "consumption_rate": self.consumption_rate,
            "imbalance_ratio": self.imbalance_ratio,
            "is_rollout_congested": self.is_rollout_congested,
            "is_train_congested": self.is_train_congested,
            "suggested_action": self.suggested_action.value,
            "confidence": self.confidence,
            "urgency": self.urgency,
        }


class CongestionMonitor:
    """
    Congestion Monitor

    Monitors rollout production rate and train consumption rate,
    calculates congestion status and provides scheduling suggestions.
    """

    def __init__(
        self,
        window_size: int = 20,
        high_watermark: float = 0.8,
        low_watermark: float = 0.3,
        production_ema_alpha: float = 0.3,
        consumption_ema_alpha: float = 0.3,
        imbalance_threshold: float = 0.2,
    ):
        self.window_size = window_size
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.imbalance_threshold = imbalance_threshold

        # Sliding window records
        self.queue_sizes: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.queue_capacities: deque = deque(maxlen=window_size)

        # Production/consumption counts per window
        self.production_counts: deque = deque(maxlen=window_size)
        self.consumption_counts: deque = deque(maxlen=window_size)

        # EMA parameters
        self.production_ema_alpha = production_ema_alpha
        self.consumption_ema_alpha = consumption_ema_alpha
        self._ema_production: Optional[float] = None
        self._ema_consumption: Optional[float] = None

        # Rate history
        self._rate_history: deque = deque(maxlen=100)

    def record(
        self,
        queue_size: int,
        queue_capacity: int,
        produced: int,
        consumed: int,
    ):
        """
        Record a sampling point

        Args:
            queue_size: Current queue size
            queue_capacity: Queue max capacity
            produced: Number of samples produced since last record
            consumed: Number of samples consumed since last record
        """
        now = time.time()

        self.queue_sizes.append(queue_size)
        self.queue_capacities.append(queue_capacity)
        self.production_counts.append(produced)
        self.consumption_counts.append(consumed)

        # Calculate time delta
        if len(self.timestamps) > 0:
            time_delta = now - self.timestamps[-1]
            if time_delta > 0:
                rate_production = produced / time_delta
                rate_consumption = consumed / time_delta

                # Update EMA
                if self._ema_production is None:
                    self._ema_production = rate_production
                else:
                    self._ema_production = (
                        self.production_ema_alpha * rate_production
                        + (1 - self.production_ema_alpha) * self._ema_production
                    )

                if self._ema_consumption is None:
                    self._ema_consumption = rate_consumption
                else:
                    self._ema_consumption = (
                        self.consumption_ema_alpha * rate_consumption
                        + (1 - self.consumption_ema_alpha) * self._ema_consumption
                    )

                # Record in history
                self._rate_history.append(
                    {
                        "timestamp": now,
                        "production_rate": rate_production,
                        "consumption_rate": rate_consumption,
                        "queue_util": queue_size / max(queue_capacity, 1),
                    }
                )

        self.timestamps.append(now)

    def get_metrics(self) -> CongestionMetrics:
        """Get current congestion metrics"""
        # Current queue utilization
        if len(self.queue_sizes) > 0:
            queue_size = self.queue_sizes[-1]
            queue_capacity = self.queue_capacities[-1]
            queue_util = queue_size / max(queue_capacity, 1)
        else:
            queue_size = 0
            queue_capacity = 1
            queue_util = 0.0

        # Get EMA rates
        prod_rate = self._ema_production or 0.0
        cons_rate = self._ema_consumption or 0.0

        # Calculate imbalance ratio
        if cons_rate > 0:
            imbalance_ratio = prod_rate / cons_rate
        else:
            imbalance_ratio = 1.0 if prod_rate > 0 else 0.0

        # Determine congestion status
        is_rollout_congested = queue_util > self.high_watermark
        is_train_congested = queue_util < self.low_watermark

        # Determine suggested action
        if prod_rate < cons_rate and queue_util > self.high_watermark:
            suggested = SuggestedAction.SCALE_UP_ROLLOUT
            confidence = min(1.0, queue_util)
            urgency = queue_util - self.high_watermark
        elif prod_rate > cons_rate and queue_util < self.low_watermark:
            suggested = SuggestedAction.SCALE_UP_TRAIN
            confidence = min(1.0, 1.0 - queue_util)
            urgency = self.low_watermark - queue_util
        elif prod_rate > cons_rate * 1.5 and queue_util > self.high_watermark:
            # Severe imbalance
            suggested = SuggestedAction.SCALE_UP_ROLLOUT
            confidence = min(1.0, imbalance_ratio - 1.0)
            urgency = min(1.0, imbalance_ratio - 1.0)
        elif cons_rate > prod_rate * 1.5 and queue_util < self.low_watermark:
            # Train is much faster
            suggested = SuggestedAction.SCALE_UP_TRAIN
            confidence = min(1.0, 1.0 / max(imbalance_ratio, 0.1) - 1.0)
            urgency = min(1.0, 1.0 / max(imbalance_ratio, 0.1) - 1.0)
        elif queue_util > 0.9:
            # Very high queue - urgent rollout scale up
            suggested = SuggestedAction.SCALE_UP_ROLLOUT
            confidence = 0.9
            urgency = 0.9
        else:
            suggested = SuggestedAction.STABLE
            confidence = 0.5
            urgency = 0.0

        return CongestionMetrics(
            timestamp=time.time(),
            queue_size=queue_size,
            queue_capacity=queue_capacity,
            queue_utilization=queue_util,
            production_rate=prod_rate,
            consumption_rate=cons_rate,
            is_rollout_congested=is_rollout_congested,
            is_train_congested=is_train_congested,
            imbalance_ratio=imbalance_ratio,
            suggested_action=suggested,
            confidence=confidence,
            urgency=urgency,
        )

    def get_rate_trend(self) -> str:
        """Get rate trend: 'increasing', 'decreasing', or 'stable'"""
        if len(self._rate_history) < 5:
            return "stable"

        recent = list(self._rate_history)[-5:]
        early_rate = sum(r["production_rate"] for r in recent[:2]) / 2
        late_rate = sum(r["production_rate"] for r in recent[-2:]) / 2

        if late_rate > early_rate * 1.1:
            return "increasing"
        elif late_rate < early_rate * 0.9:
            return "decreasing"
        return "stable"

    def reset(self):
        """Reset all metrics"""
        self.queue_sizes.clear()
        self.timestamps.clear()
        self.queue_capacities.clear()
        self.production_counts.clear()
        self.consumption_counts.clear()
        self._ema_production = None
        self._ema_consumption = None
        self._rate_history.clear()


class ResourceCoordinator:
    """
    Resource Scheduling Coordinator

    Coordinates elastic resource allocation between rollout and train,
    balancing production and consumption rates.
    """

    def __init__(
        self,
        resource_manager: "ElasticResourceManager",
        congestion_monitor: CongestionMonitor,
        sync_trigger_interval: int = 4,  # Trigger sync every N steps
        auto_scale_enabled: bool = True,
        min_elastic_for_rollout: int = 0,
        min_elastic_for_train: int = 0,
    ):
        self.resource_manager = resource_manager
        self.congestion_monitor = congestion_monitor

        # Configuration
        self.sync_trigger_interval = sync_trigger_interval
        self.auto_scale_enabled = auto_scale_enabled
        self.min_elastic_for_rollout = min_elastic_for_rollout
        self.min_elastic_for_train = min_elastic_for_train

        # Coordination state
        self.current_step = 0
        self.is_coordinating = False
        self._pending_actions: list = []
        self._last_metrics: Optional[CongestionMetrics] = None

        # Callbacks
        self.on_switch_complete: Optional[Callable] = None
        self.on_sync_triggered: Optional[Callable] = None
        self.on_metrics_logged: Optional[Callable] = None

        # Statistics
        self.stats = {
            "total_coordination_steps": 0,
            "rollout_scale_ups": 0,
            "train_scale_ups": 0,
            "syncs_triggered": 0,
        }

    async def coordinate(self) -> Optional[CongestionMetrics]:
        """
        Execute one round of coordination

        Flow:
        1. Get current congestion status
        2. Decide whether to switch resources based on status
        3. Execute switches
        4. Trigger parameter sync if needed
        """
        self.is_coordinating = True

        try:
            # 1. Get metrics
            metrics = self.congestion_monitor.get_metrics()
            self._last_metrics = metrics

            # Log metrics
            if self.on_metrics_logged:
                self.on_metrics_logged(metrics)
            else:
                logger.debug(f"Congestion metrics: {metrics.to_dict()}")

            # 2. Make decision based on metrics
            if self.auto_scale_enabled and metrics.confidence > 0.6:
                if metrics.suggested_action == SuggestedAction.SCALE_UP_ROLLOUT:
                    await self._scale_up_rollout(metrics.confidence)
                elif metrics.suggested_action == SuggestedAction.SCALE_UP_TRAIN:
                    await self._scale_up_train(metrics.confidence)

            # 3. Check if sync should be triggered
            if self.current_step % self.sync_trigger_interval == 0:
                await self._trigger_parameter_sync()

            self.current_step += 1
            self.stats["total_coordination_steps"] += 1

            return metrics

        finally:
            self.is_coordinating = False

    async def _scale_up_rollout(self, confidence: float):
        """Scale up rollout resources"""
        # Calculate how many to switch
        elastic_total = len(self.resource_manager.elastic_resources)
        elastic_in_train = sum(
            1 for r in self.resource_manager.elastic_resources if r.current_mode == HybridEngineMode.TRAIN
        )
        available = elastic_total - max(elastic_in_train, self.min_elastic_for_train)

        if available <= 0:
            logger.debug("No elastic resources available for rollout")
            return

        # Determine number to switch based on confidence
        n_to_switch = max(1, min(int(confidence * available) + 1, available))

        # Execute switch
        switched = await self.resource_manager.switch_elastic_to_rollout(
            n_resources=n_to_switch,
            sync_callback=self._sync_before_switch,
        )

        if switched:
            self.stats["rollout_scale_ups"] += len(switched)
            logger.info(f"Scaled up rollout by {len(switched)} elastic resources")

            if self.on_switch_complete:
                await self.on_switch_complete("rollout", switched)

    async def _scale_up_train(self, confidence: float):
        """Scale up train resources"""
        # Calculate how many to switch
        elastic_total = len(self.resource_manager.elastic_resources)
        elastic_in_rollout = sum(
            1 for r in self.resource_manager.elastic_resources if r.current_mode == HybridEngineMode.ROLLOUT
        )
        available = elastic_total - max(elastic_in_rollout, self.min_elastic_for_rollout)

        if available <= 0:
            logger.debug("No elastic resources available for train")
            return

        # Determine number to switch based on confidence
        n_to_switch = max(1, min(int(confidence * available) + 1, available))

        # Execute switch
        switched = await self.resource_manager.switch_elastic_to_train(
            n_resources=n_to_switch,
            sync_callback=self._sync_before_switch,
        )

        if switched:
            self.stats["train_scale_ups"] += len(switched)
            logger.info(f"Scaled up train by {len(switched)} elastic resources")

            if self.on_switch_complete:
                await self.on_switch_complete("train", switched)

    async def _sync_before_switch(self, resource: "HybridEngineResource"):
        """Parameter sync callback before switching"""
        if self.on_sync_triggered:
            await self.on_sync_triggered([resource])

    async def _trigger_parameter_sync(self):
        """Trigger parameter sync"""
        # Get all currently active rollout resources
        rollout_resources = self.resource_manager.get_rollout_resources_all()

        if self.on_sync_triggered and rollout_resources:
            await self.on_sync_triggered(rollout_resources)
            self.stats["syncs_triggered"] += 1

    def get_current_status(self) -> dict:
        """Get current coordination status"""
        return {
            "step": self.current_step,
            "is_coordinating": self.is_coordinating,
            "last_metrics": self._last_metrics.to_dict() if self._last_metrics else None,
            "stats": self.stats.copy(),
            "resource_status": self.resource_manager.get_status_summary(),
        }

    async def force_sync(self):
        """Force an immediate parameter sync"""
        await self._trigger_parameter_sync()


class CoordinatorLoop:
    """
    Main coordination loop

    Runs the coordination loop continuously with configurable intervals.
    """

    def __init__(
        self,
        coordinator: ResourceCoordinator,
        check_interval: float = 1.0,  # Check interval in seconds
    ):
        self.coordinator = coordinator
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the coordination loop"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Coordinator loop started")

    async def stop(self):
        """Stop the coordination loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Coordinator loop stopped")

    async def _run_loop(self):
        """Main loop"""
        while self._running:
            try:
                await self.coordinator.coordinate()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in coordination loop: {e}")
                await asyncio.sleep(self.check_interval)
