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

"""ReplayBuffer: Ray Actor for metadata channel + slot-based flow control in TQ fully async training.

Architecture:
- TQFullyAsyncRollouter: Calls acquire_slot() in _feed_samples, writes results to TQ with status=finish
- TQFullyAsyncTrainer: Consumes finished samples via wait_and_sample(), reads data from TQ

Status Flow:
    (Rollouter writes status=finish) -> finish -> (Trainer consumes & removes)

Slot Control:
- acquire_slot(): Rollouter calls in _feed_samples BEFORE putting to pending_queue (blocking)
- release_slot(): Auto-called when poll thread detects status=finish transition
- This replaces _should_pause_generation() + MessageQueue.queue_size backpressure

Usage:
    from verl.experimental.fully_async_policy_tq.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.remote(max_pending_slots=256)
    # Rollouter side:
    acquired = await asyncio.wrap_future(rb.acquire_slot.remote(timeout=None).future())
    # Trainer side:
    sampled = await asyncio.wrap_future(rb.wait_and_sample.remote("train", batch_size=64).future())
"""

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import ray

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Status type
StatusType = Literal["pending", "running", "partial", "finish"]


@ray.remote
class ReplayBuffer:
    """Ray Actor: metadata channel + slot-based flow control for TQ fully async training.

    Replaces MessageQueue (data channel) in the original fully_async_policy.
    Key responsibilities:
    1. Slot-based backpressure: acquire_slot() blocks rollouter at dataloader source
    2. Metadata storage: tracks status of each sample via TQ kv_list polling
    3. Consumer interface: wait_and_sample() for trainer to get finished samples
    4. Version tracking: reset_staleness() for parameter sync coordination
    """

    def __init__(self, max_pending_slots: int = 256, poll_interval: float = 1.0):
        # Partition -> {key: tags_dict}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.lock = threading.Lock()
        self.poll_interval = poll_interval
        self._finished = False

        # Slot control for TQ request count (replaces MessageQueue.queue_size)
        self.max_pending_slots = max_pending_slots
        self._pending_slots = 0  # acquired but not yet finish
        self._slot_available = threading.Condition(self.lock)

        # Background thread: sync meta from TQ kv_list
        self._poll_thread = threading.Thread(target=self._poll_from_tq, daemon=True)
        self._poll_thread.start()

        # Background thread: periodic statistics logging
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        print(f"[ReplayBuffer] initialized with max_pending_slots={max_pending_slots}, poll_interval={poll_interval}")

    def _poll_from_tq(self):
        """Background thread that polls TQ for metadata updates.

        Auto-releases slots when entries transition to 'finish'.
        This is the core mechanism that unblocks acquire_slot() in _feed_samples.
        """
        try:
            while True:
                data = tq.kv_list()
                if data is not None:
                    for partition_id, items in data.items():
                        with self.lock:
                            for key, meta in items.items():
                                prev_meta = self.partitions.get(partition_id, {}).get(key, {})
                                prev_status = prev_meta.get("current_status")
                                new_status = meta.get("current_status")

                                # Update metadata
                                self.partitions[partition_id][key] = meta

                                # Detect status transitions: * -> finish => release slot
                                if new_status == "finish" and prev_status != "finish":
                                    self._pending_slots = max(0, self._pending_slots - 1)
                                    self._slot_available.notify()
                                    logger.debug(
                                        f"[ReplayBuffer] Slot released for {key} "
                                        f"({prev_status}->finish), "
                                        f"pending_slots={self._pending_slots}"
                                    )

                time.sleep(self.poll_interval)
        except Exception as e:
            logger.error(f"[ReplayBuffer] _poll_from_tq error: {e}")
            os._exit(1)

    def _monitor_loop(self):
        """Background thread that periodically logs buffer statistics."""
        from pprint import pformat

        monitor_interval = 60.0
        while not self._finished:
            time.sleep(monitor_interval)
            if self._finished:
                break
            try:
                stats = self.get_statistics()
                print(f"[ReplayBuffer][Monitor] {pformat(stats)}")
            except Exception as e:
                logger.error(f"[ReplayBuffer] _monitor_loop error: {e}")

    # ======== Slot control (backpressure — called by Rollouter _feed_samples) ========

    def acquire_slot(self, timeout: float | None = None) -> bool:
        """Acquire a slot before processing a dataloader sample.

        Called by TQFullyAsyncRollouter in _feed_samples(), BEFORE putting
        the sample into pending_queue. This implements source-level flow control:
        when too many samples are in-flight, the dataloader blocks here.

        Args:
            timeout: Max seconds to wait. None = block indefinitely.

        Returns:
            True if slot acquired, False if timed out or finished.
        """
        with self._slot_available:
            while self._pending_slots >= self.max_pending_slots:
                if self._finished:
                    return False
                if timeout is not None:
                    if not self._slot_available.wait(timeout):
                        return False
                else:
                    self._slot_available.wait()
                    if self._finished:
                        return False
            self._pending_slots += 1
            return True

    def release_slot(self):
        """Manually release a slot (e.g., on error/drop).

        Normally slots are auto-released by _poll_from_tq when it detects
        status transition to 'finish'.
        """
        with self._slot_available:
            self._pending_slots = max(0, self._pending_slots - 1)
            self._slot_available.notify()

    @property
    def pending_slots(self) -> int:
        """Current number of acquired (in-flight) slots."""
        with self.lock:
            return self._pending_slots

    @property
    def available_slots(self) -> int:
        """Number of slots available for new requests."""
        with self.lock:
            return max(0, self.max_pending_slots - self._pending_slots)

    # ======== Consumer interface (called by TQFullyAsyncTrainer) ========

    def wait_and_sample(
        self,
        partition_id: str,
        batch_size: int,
    ) -> list[tuple[str, dict]] | None:
        """Block until enough finish samples are ready or finish signal received.

        Called by TQFullyAsyncTrainer at the start of each training step.
        Returns keys whose metadata has current_status='finish'.

        Args:
            partition_id: Partition to sample from, e.g. "train".
            batch_size: Desired number of samples.

        Returns:
            List of (key, meta) tuples. Length may be < batch_size when finished.
            Returns None if finished and no samples available.
        """
        while True:
            with self.lock:
                part = self.partitions.get(partition_id)
                if part is None:
                    # Partition not yet created (no samples written yet), keep waiting
                    pass
                else:
                    ready = [(k, v) for k, v in part.items() if v.get("current_status") == "finish"]

                    if len(ready) >= batch_size:
                        return ready[:batch_size]

                    if self._finished:
                        return ready if ready else None

            time.sleep(self.poll_interval)

    def remove(self, partition_id: str, keys: list[str]):
        """Remove sampled metadata after training consumption.

        Called by TQFullyAsyncTrainer after finishing a training step.
        """
        with self.lock:
            part = self.partitions.get(partition_id, {})
            for k in keys:
                part.pop(k, None)

    # ======== Control signals ========

    def signal_finish(self):
        """Signal that production is complete (no more samples will arrive).

        Called by TQFullyAsyncRollouter after _feed_samples finishes.
        """
        with self._slot_available:
            self._finished = True
            self._slot_available.notify_all()

    def is_finished(self) -> bool:
        """True if finish signaled."""
        with self.lock:
            return self._finished

    # ======== Staleness / version tracking (called by Trainer after param sync) ========

    def reset_staleness(self, active_task_count: int = 0) -> dict:
        """Reset staleness after parameter update.

        Called by TQFullyAsyncTrainer after checkpoint_manager.update_weights().
        Increments current model version and computes timing metrics.

        Args:
            active_task_count: Number of currently active tasks in the rollouter
                              (tasks that are in-flight but not yet written to TQ as finish).

        Returns:
            Dict with timing metrics for logging.
        """
        with self.lock:
            # Count ready (finish) samples in RB
            ready_count = self.count_by_status("finish")
            running_count = self.count_by_status("running") + self.count_by_status("partial")
            pending_count = self.count_by_status("pending")

            self._staleness_samples = active_task_count + ready_count + running_count + pending_count

            # Compute timing metrics
            now = time.time()
            if not hasattr(self, "_last_reset_time"):
                self._last_reset_time = now
            version_time = max(now - self._last_reset_time, 1e-6)

            if hasattr(self, "_idle_start_time") and self._idle_start_time > self._last_reset_time:
                active_time = self._idle_start_time - self._last_reset_time
                idle_ratio = 1 - active_time / version_time
            else:
                active_time = version_time
                idle_ratio = 0

            self._current_model_version = getattr(self, "_current_model_version", 0) + 1
            self._last_reset_time = now

            timing_raw = {
                "fully_async/rollouter/active_time": active_time,
                "fully_async/rollouter/version_time": version_time,
                "fully_async/rollouter/idle_ratio": idle_ratio,
            }

            print(
                f"[ReplayBuffer][reset_staleness] "
                f"model_version={self._current_model_version}, "
                f"staleness_samples={self._staleness_samples}, "
                f"ready={ready_count}, running={running_count}, pending={pending_count}, "
                f"active_tasks={active_task_count}, "
                f"idle_ratio={idle_ratio:.4f}"
            )

        return timing_raw

    def note_idle_start(self):
        """Record that rollouter has gone idle (for idle_ratio calculation)."""
        with self.lock:
            self._idle_start_time = time.time()

    @property
    def current_model_version(self) -> int:
        """Current model version (incremented on each reset_staleness call)."""
        with self.lock:
            return getattr(self, "_current_model_version", 0)

    @property
    def staleness_samples(self) -> int:
        """Current staleness sample count (set by last reset_staleness call)."""
        with self.lock:
            return getattr(self, "_staleness_samples", 0)

    # ======== Statistics ========

    def count_by_status(self, status: StatusType, partition_id: str | None = None) -> int:
        """Count samples by status."""
        with self.lock:
            if partition_id:
                parts = [self.partitions.get(partition_id, {})]
            else:
                parts = list(self.partitions.values())
            return sum(1 for part in parts for v in part.values() if v.get("current_status") == status)

    def pending_count(self) -> int:
        """Count of samples with current_status='pending'."""
        return self.count_by_status("pending")

    def running_count(self) -> int:
        """Count of samples with current_status in ['running', 'partial']."""
        return self.count_by_status("running") + self.count_by_status("partial")

    def ready_count(self) -> int:
        """Count of samples with current_status='finish' available for consumption."""
        return self.count_by_status("finish")

    @property
    def total_in_flight(self) -> int:
        """Total in-flight samples across all non-terminal states."""
        with self.lock:
            return (
                self._pending_slots
                + self.count_by_status("pending")
                + self.count_by_status("running")
                + self.count_by_status("partial")
                + self.count_by_status("finish")
            )

    def get_staleness_statistics(self, current_version: int, partition_id: str = "train") -> dict[str, Any]:
        """Calculate staleness distribution for monitoring.

        Args:
            current_version: Current parameter version on Trainer.
            partition_id: Partition to analyze.

        Returns:
            Dict with staleness statistics (mean, max, min, count).
        """
        with self.lock:
            part = self.partitions.get(partition_id, {})
            version_spans = []

            for meta in part.values():
                if meta.get("current_status") == "finish":
                    start_version = meta.get("start_model_version", current_version)
                    span = current_version - start_version
                    version_spans.append(span)

            if not version_spans:
                return {
                    "staleness/mean": 0,
                    "staleness/max": 0,
                    "staleness/min": 0,
                    "staleness/count": 0,
                }

            return {
                "staleness/mean": float(np.mean(version_spans)),
                "staleness/max": int(max(version_spans)),
                "staleness/min": int(min(version_spans)),
                "staleness/count": len(version_spans),
            }

    def get_version_distribution(self, partition_id: str = "train") -> dict[str, int]:
        """Get distribution of samples across version spans.

        Returns:
            Dict mapping version_span -> count.
        """
        from collections import defaultdict

        with self.lock:
            part = self.partitions.get(partition_id, {})
            distribution = defaultdict(int)

            for meta in part.values():
                if meta.get("current_status") == "finish":
                    start_version = meta.get("start_model_version")
                    end_version = meta.get("end_model_version")
                    if start_version is not None and end_version is not None:
                        span = end_version - start_version
                        distribution[f"span_{span}"] += 1

            return dict(distribution)

    def get_statistics(self) -> dict:
        """Return statistics about the buffer state."""
        with self.lock:
            partition_stats = {}
            for pid, part in self.partitions.items():
                stats = {"pending": 0, "running": 0, "partial": 0, "finish": 0, "total": len(part)}
                for v in part.values():
                    status = v.get("current_status", "unknown")
                    if status in stats:
                        stats[status] += 1
                partition_stats[pid] = stats

            return {
                "partitions": partition_stats,
                "total_pending": self.count_by_status("pending"),
                "total_running": self.count_by_status("running") + self.count_by_status("partial"),
                "total_ready": self.count_by_status("finish"),
                "total_in_flight": self.total_in_flight,
                "pending_slots": self._pending_slots,
                "max_pending_slots": self.max_pending_slots,
                "available_slots": max(0, self.max_pending_slots - self._pending_slots),
                "finished": self._finished,
            }
