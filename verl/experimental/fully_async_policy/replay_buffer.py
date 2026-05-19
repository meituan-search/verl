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


@ray.remote(max_concurrency=100)
class ReplayBuffer:
    """Ray Actor: metadata channel + slot-based flow control for TQ fully async training.

    Replaces MessageQueue (data channel) in the original fully_async_policy.
    Key responsibilities:
    1. Slot-based backpressure: acquire_slot() blocks rollouter at dataloader source
    2. Metadata storage: tracks status of each sample via TQ kv_list polling
    3. Consumer interface: wait_and_sample() for trainer to get finished samples
    4. Version tracking: reset_staleness() for parameter sync coordination
    """

    def __init__(
        self,
        max_version_slots: int,
        max_pending_slots: int = 256,
        poll_interval: float = 1.0,
    ):
        # Partition -> {key: tags_dict}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.lock = threading.Lock()
        self.poll_interval = poll_interval
        self._finished = False

        # ======== Layer 1: Physical slot control (concurrency / OOM guard) ========
        # Limits simultaneous in-flight samples.
        # Acquired in _feed_samples (Rollouter), released on finish (mark_finish / poll).
        # Maps to: max_concurrent_samples (e.g. TP * PP * 16)
        self.max_pending_slots = max_pending_slots
        self._pending_slots = 0  # acquired but not yet finish
        self._slot_available = threading.Condition(self.lock)

        # ======== Layer 2: Version window control (staleness guard) ========
        # Limits total slots issued per model version.
        # When _version_slots >= max_version_slots, acquire_slot() blocks until
        # reset_staleness() is called (after param sync).
        # Maps to: max_required_samples (e.g. required_samples * trigger_parameter_sync_step)
        self.max_version_slots = max_version_slots
        self._version_slots = 0  # cumulative slots issued in current version
        self._current_model_version = 0

        # Background threads — lazy start via acquire_slot()
        self._poll_thread = None
        self._monitor_thread = None
        self._poll_started = False

        # Initialize TQ in this actor process so _poll_from_tq can call tq.kv_list()
        try:
            import transfer_queue as tq

            tq.init()
            print("[ReplayBuffer] TQ initialized in RB actor process", flush=True)
        except Exception as e:
            print(f"[ReplayBuffer] TQ init warning: {e}", flush=True)

        print(
            f"[ReplayBuffer] initialized with "
            f"max_pending_slots={max_pending_slots}, "
            f"max_version_slots={max_version_slots}, "
            f"poll_interval={poll_interval}"
        )

    def ensure_polling_started(self):
        """Lazily start background threads after TQ is initialized.

        Called from acquire_slot (first call from rollouter _feed_samples)
        which happens after tq.init() in set_max_required_samples / fit().
        """
        with self.lock:
            if self._poll_started:
                return
            self._poll_started = True

        self._poll_thread = threading.Thread(target=self._poll_from_tq, daemon=True)
        self._poll_thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("[ReplayBuffer] Background polling & monitor threads started")

    def _poll_from_tq(self):
        """Background thread that polls TQ for metadata updates.

        Auto-releases slots when entries transition to 'finish'.
        This is the core mechanism that unblocks acquire_slot() in _feed_samples.
        """
        poll_count = 0
        try:
            while True:
                data = tq.kv_list()
                poll_count += 1
                if poll_count % 10 == 1:  # Log every 10 polls (~10s with 1s interval)
                    print(f"[ReplayBuffer][_poll] poll #{poll_count}, data={data}", flush=True)
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
                                    pd_slots = self._pending_slots
                                    print(
                                        f"[RB] Slot released for {key} ({prev_status}->finish), pending={pd_slots}",
                                        flush=True,
                                    )

                time.sleep(self.poll_interval)
        except Exception as e:
            print(f"[ReplayBuffer] _poll_from_tq error: {e}", flush=True)
            import traceback

            traceback.print_exc()
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
        the sample into pending_queue. This implements **dual-layer flow control**:

        Layer 1 (Physical): ``_pending_slots < max_pending_slots``
            Limits simultaneous in-flight samples to prevent OOM / GPU overload.
            Slot is released when sample reaches status=finish (mark_finish or poll).

        Layer 2 (Version window): ``_version_slots < max_version_slots``
            Limits total slots issued per model version to control staleness.
            When the version window is full, acquire_slot() blocks until
            reset_staleness() is called after parameter synchronization.
            Only enforced if max_version_slots is set (via set_version_config()).

        Both conditions must be satisfied for a slot to be issued.

        Args:
            timeout: Max seconds to wait. None = block indefinitely.

        Returns:
            True if slot acquired, False if timed out or finished.
        """
        # Lazy-start background polling threads on first call (without lock)
        if not getattr(self, "_poll_started", False):
            self._poll_started = True
            self._poll_thread = threading.Thread(target=self._poll_from_tq, daemon=True)
            self._poll_thread.start()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            print("[ReplayBuffer] Background polling & monitor threads started", flush=True)

        # Lock-free dual-condition slot acquisition with poll loop
        import time as _time

        _wait_forever = timeout is None
        _deadline_abs = None if _wait_forever else (_time.monotonic() + timeout)
        _poll_interval = 0.1

        while True:
            # Lock-free check (safe due to Ray actor serialization + GIL)
            if self._finished:
                return False

            # Layer 1: Physical concurrency limit
            physical_ok = self._pending_slots < self.max_pending_slots

            # Layer 2: Version window limit (staleness control)
            # If max_version_slots is not yet set (None), skip this check
            version_ok = self.max_version_slots is None or self._version_slots < self.max_version_slots

            if physical_ok and version_ok:
                self._pending_slots += 1
                self._version_slots += 1
                return True

            # Diagnostic: log which condition blocked us (only once per block)
            if not physical_ok:
                print(
                    f"[RB][acquire_slot] BLOCKED: physical full "
                    f"(pending={self._pending_slots}/{self.max_pending_slots})",
                    flush=True,
                )
            elif not version_ok:
                print(
                    f"[RB][acquire_slot] BLOCKED: version window full "
                    f"(version_slots={self._version_slots}/{self.max_version_slots}, "
                    f"version={self._current_model_version}) — waiting for reset_staleness",
                    flush=True,
                )

            # Wait before retrying
            if not _wait_forever:
                remaining = _deadline_abs - _time.monotonic()
                if remaining <= 0:
                    return False
                _time.sleep(min(_poll_interval, remaining))
            else:
                _time.sleep(_poll_interval)

    def release_slot(self):
        """Manually release a slot (e.g., on error/drop).

        Normally slots are auto-released by mark_finish() when rollouter
        calls it after writing to TQ.
        """
        self._pending_slots = max(0, self._pending_slots - 1)
        print(f"[ReplayBuffer][release_slot] pending={self._pending_slots}", flush=True)

    def mark_finish(self, key: str, partition_id: str = "train", meta: dict | None = None):
        """Mark a sample as finished (called by Rollouter after TQ write).

        This replaces the _poll_from_tq mechanism for cross-process scenarios
        where RB's tq.kv_list() cannot see data written by Rollouter's TQ instance.

        Args:
            key: Sample key (e.g. "sample_0_42")
            partition_id: Partition name
            meta: Optional metadata dict (tags)
        """
        if meta is None:
            meta = {"current_status": "finish"}
        self.partitions[partition_id][key] = meta
        self._pending_slots = max(0, self._pending_slots - 1)
        print(f"[ReplayBuffer][mark_finish] {key} in {partition_id}, pending={self._pending_slots}", flush=True)

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
        """Block until enough finish samples are ready or production is fully complete.

        Called by TQFullyAsyncTrainer at the start of each training step.
        Returns keys whose metadata has current_status='finish'.

        Uses lock-free reads (safe due to Ray actor serialization + GIL for simple ops).
        mark_finish() is the writer and does not use lock either.

        IMPORTANT — Termination semantics:
        -----------------------------------
        signal_finish() is called by Rollouter's _streaming_generation_main()
        only AFTER both:
          1. _feed_samples() has finished iterating the dataloader AND
          2. _processor_worker() has processed ALL samples from pending_queue
             (every sample has been inferred, written to TQ, and mark_finish()'d)

        In other words, signal_finish() means "production is FULLY done — every
        sample that entered the pipeline has completed inference and been written
        to TQ with status=finish". There are NO in-flight samples remaining when
        this fires.

        Therefore, once _finished=True, it is safe to return remaining samples
        or None immediately — no need to check _pending_slots because they are
        guaranteed to be 0 at this point.

        Decision matrix:
        ┌─────────────┬───────────────┬──────────────────────────────────┐
        │ _finished   │ ready count   │ Action                           │
        ├─────────────┼───────────────┼──────────────────────────────────┤
        │ False       │ ≥ batch_size  │ Return batch (normal path)       │
        │ False       │ < batch_size  │ Keep waiting                      │
        │ True        │ > 0           │ Return remaining (< batch_size)   │
        │ True        │ = 0           │ Return None (truly done)          │
        └─────────────┴───────────────┴──────────────────────────────────┘

        Args:
            partition_id: Partition to sample from, e.g. "train".
            batch_size: Desired number of samples.

        Returns:
            List of (key, meta) tuples. Length may be < batch_size when finished.
            Returns None if finished and no samples available.
        """
        # Lazy-start background polling threads (without lock)
        if not getattr(self, "_poll_started", False):
            self._poll_started = True
            self._poll_thread = threading.Thread(target=self._poll_from_tq, daemon=True)
            self._poll_thread.start()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

        while True:
            # Lock-free read (safe: Ray serializes actor calls, GIL protects dict/int ops)
            part = self.partitions.get(partition_id)
            if part is not None:
                ready = [(k, v) for k, v in part.items() if v.get("current_status") == "finish"]

                if len(ready) >= batch_size:
                    print(
                        f"[RB][wait_and_sample] Returning {len(ready[:batch_size])} samples from {partition_id}",
                        flush=True,
                    )
                    return ready[:batch_size]

                # Production is fully complete — safe to drain remaining or return None
                if self._finished:
                    if ready:
                        print(
                            "[ReplayBuffer][wait_and_sample] Finished, returning "
                            f"{len(ready)} remaining samples (partial batch)",
                            flush=True,
                        )
                        return ready
                    else:
                        print(
                            "[ReplayBuffer][wait_and_sample] Finished, no remaining samples, returning None",
                            flush=True,
                        )
                        return None

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
        """Signal that production is fully complete — all samples are done.

        Called by TQFullyAsyncRollouter._streaming_generation_main() in its
        finally block, which runs ONLY after both:
          1. _feed_samples() has exhausted the dataloader (put None sentinel), AND
          2. _processor_worker() has drained pending_queue completely
             (all samples inferred, written to TQ, mark_finish()'d).

        This is the definitive "no more data ever" signal. When this fires,
        wait_and_sample() can safely return remaining samples or None.
        There are guaranteed to be zero in-flight / un-finished samples.
        """
        with self._slot_available:
            self._finished = True
            self._slot_available.notify_all()

    def is_finished(self) -> bool:
        """True if finish signaled."""
        with self.lock:
            return self._finished

    # ======== Version window control (called by Trainer after param sync) ========

    def reset_staleness(self) -> dict:
        """Reset the version window after parameter synchronization.

        Called by TQFullyAsyncTrainer (via Rollouter) after
        checkpoint_manager.update_weights(). This:

        1. Resets ``_version_slots = 0`` — unblocks acquire_slot() if it was
           blocked on the version window being full.
        2. Increments ``_current_model_version += 1``.

        Returns:
            Dict with timing/metrics for logging.
        """
        now = time.time()
        with self.lock:
            prev_version = self._current_model_version
            prev_version_slots = self._version_slots

            # Core: reset version window to unblock acquire_slot()
            self._version_slots = 0
            self._current_model_version += 1

            # Timing metrics
            if not hasattr(self, "_last_reset_time"):
                self._last_reset_time = now
            version_time = max(now - self._last_reset_time, 1e-6)

            if hasattr(self, "_idle_start_time") and self._idle_start_time > self._last_reset_time:
                active_time = self._idle_start_time - self._last_reset_time
                idle_ratio = 1 - active_time / version_time
            else:
                active_time = version_time
                idle_ratio = 0

            self._last_reset_time = now

            timing_raw = {
                "fully_async/rollouter/active_time": active_time,
                "fully_async/rollouter/version_time": version_time,
                "fully_async/rollouter/idle_ratio": idle_ratio,
            }

            print(
                f"[ReplayBuffer][reset_staleness] "
                f"version: {prev_version} -> {self._current_model_version}, "
                f"version_slots: {prev_version_slots} -> 0, "
                f"pending_slots: {self._pending_slots}, "
                f"idle_ratio: {idle_ratio:.4f}",
                flush=True,
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
            return self._current_model_version

    # ======== Statistics ========

    def count_by_status(self, status: StatusType, partition_id: str | None = None) -> int:
        """Count samples by status.

        NOTE: This is called both from inside lock-holding contexts (e.g. reset_staleness)
        and from outside. Since threading.Lock is NOT reentrant, we must NOT acquire
        the lock here if we're already holding it. The caller is responsible for
        acquiring the lock when needed.
        """
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
        """Total in-flight samples across all non-terminal states.

        NOTE: Caller must hold self.lock if thread-safety is required.
        """
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
                # Layer 1: Physical slot control
                "pending_slots": self._pending_slots,
                "max_pending_slots": self.max_pending_slots,
                "available_physical_slots": max(0, self.max_pending_slots - self._pending_slots),
                # Layer 2: Version window control
                "version_slots": self._version_slots,
                "max_version_slots": self.max_version_slots,
                "available_version_slots": max(0, (self.max_version_slots or 0) - self._version_slots),
                "current_model_version": self._current_model_version,
                # Control
                "finished": self._finished,
            }
