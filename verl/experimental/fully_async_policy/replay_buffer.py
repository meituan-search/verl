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
- TQFullyAsyncRollouter: Calls acquire_slot() in _feed_samples, writes results to TQ with status=finish,
  then calls release_slot() to release the slot after successful TQ write
- TQFullyAsyncTrainer: Consumes finished samples via wait_and_sample(), reads data from TQ

Status Flow:
    (Rollouter writes status=finish) -> finish -> (Trainer consumes & removes)

Slot Control:
- acquire_slot(): Rollouter calls in _feed_samples BEFORE putting to pending_queue (blocking)
- release_slot(): Called by Rollouter after successfully writing sample to TQ (normal path),
  or on error/drop path (sample never written to TQ)
- This replaces _should_pause_generation() + MessageQueue.queue_size backpressure

Usage:
    from verl.experimental.fully_async_policy_tq.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.remote(max_pending_slots=256)
    # Rollouter side:
    acquired = await asyncio.wrap_future(rb.acquire_slot.remote(timeout=None).future())
    # ... write to TQ ...
    await rb.release_slot.remote()  # release after successful TQ write
    # Trainer side:
    sampled = await asyncio.wrap_future(rb.wait_and_sample.remote("train", batch_size=64).future())
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from pprint import pformat
from typing import Literal

import ray

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

from verl.experimental.fully_async_policy.detach_utils import safe_create_task

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
    2. Metadata storage: tracks status of each sample (updated by caller via update_metadata)
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
        self._idle_start_time = None
        self._last_reset_time = None

        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.poll_interval = poll_interval
        self._finished = False

        # ======== Layer 1: Physical slot control (concurrency / OOM guard) ========
        # Limits simultaneous in-flight samples.
        # Acquired in _feed_samples (Rollouter), released by release_slot() after TQ write.
        # Maps to: max_concurrent_samples (e.g. TP * PP * 16)
        self.max_pending_slots = max_pending_slots
        self._pending_slots = 0  # acquired but not yet finish
        # Condition for slot flow control: acquire_slot waits, release_slot/reset_staleness/signal_finish notify
        self._slot_available = asyncio.Condition()
        # Condition for data availability: wait_and_sample waits, _poll_from_tq/signal_finish notify
        self._data_available = asyncio.Condition()

        # ======== Layer 2: Version window control (staleness guard) ========
        # Limits total slots issued per model version.
        # When _version_slots >= max_version_slots, acquire_slot() blocks until
        # reset_staleness() is called (after param sync).
        # Maps to: max_required_samples (e.g. required_samples * trigger_parameter_sync_step)
        self.max_version_slots = max_version_slots
        self._version_slots = 0  # cumulative slots issued in current version

        print(
            f"[ReplayBuffer] initialized with "
            f"max_pending_slots={max_pending_slots}, "
            f"max_version_slots={max_version_slots}, "
            f"poll_interval={poll_interval}"
        )

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

        # Start background tasks immediately
        self._poll_task = safe_create_task(self._poll_from_tq(), name="poll_tq_task")
        self._monitor_task = safe_create_task(self._monitor_loop(), name="monitor_task")
        print("[ReplayBuffer] Background poll & monitor tasks started (asyncio)", flush=True)

    async def _poll_from_tq(self):
        """Background asyncio task that polls TQ for metadata updates.

        Syncs TQ metadata into self.partitions so wait_and_sample() can discover
        finished samples. Slot release is NOT done here — it is handled explicitly
        by the Rollouter calling release_slot() after writing to TQ.
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
                        async with self._data_available:
                            for key, meta in items.items():
                                # Update metadata (purely for wait_and_sample discovery)
                                self.partitions[partition_id][key] = meta
                            # Notify wait_and_sample that new data may be available
                            self._data_available.notify_all()
                await asyncio.sleep(self.poll_interval)
        except Exception as e:
            print(f"[ReplayBuffer] _poll_from_tq error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            os._exit(1)

    async def _monitor_loop(self):
        """Background asyncio task that periodically logs buffer statistics."""

        monitor_interval = 60.0
        while not self._finished:
            await asyncio.sleep(monitor_interval)
            if self._finished:
                break
            try:
                stats = self.get_statistics()
                print(f"[ReplayBuffer][Monitor] {pformat(stats)}")
            except Exception as e:
                logger.error(f"[ReplayBuffer] _monitor_loop error: {e}")

    # ======== Public API ========

    async def acquire_slot(self, timeout: float | None = None) -> bool:
        """Acquire a slot before processing a dataloader sample.

        Called by TQFullyAsyncRollouter in _feed_samples(), BEFORE putting
        the sample into pending_queue. This implements **dual-layer flow control**:

        Layer 1 (Physical): ``_pending_slots < max_pending_slots``
            Limits simultaneous in-flight samples to prevent OOM / GPU overload.
            Slot is released by calling release_slot() after writing to TQ.

        Layer 2 (Version window): ``_version_slots < max_version_slots``
            Limits total slots issued per model version to control staleness.
            When the version window is full, acquire_slot() blocks until
            reset_staleness() is called after parameter synchronization.
            Only enforced if max_version_slots is set (via set_version_config()).

        Both conditions must be satisfied for a slot to be issued.

        Uses asyncio.Condition for efficient notification-based waiting instead of polling.

        Args:
            timeout: Max seconds to wait. None = block indefinitely.

        Returns:
            True if slot acquired, False if timed out or finished.
        """
        _wait_forever = timeout is None
        _deadline_abs = None if _wait_forever else (asyncio.get_event_loop().time() + timeout)

        async with self._slot_available:
            while True:
                # Check termination first
                if self._finished:
                    return False

                # Layer 1: Physical concurrency limit
                physical_ok = self._pending_slots < self.max_pending_slots

                # Layer 2: Version window limit (staleness control)
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
                    )
                elif not version_ok:
                    print(
                        f"[RB][acquire_slot] BLOCKED: version window full "
                        f"(version_slots={self._version_slots}/{self.max_version_slots}, ",
                    )

                # Wait for notification (slot release, reset_staleness, or signal_finish)
                if not _wait_forever:
                    remaining = _deadline_abs - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        return False
                    try:
                        await asyncio.wait_for(
                            self._slot_available.wait(),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        return False
                else:
                    await self._slot_available.wait()

    async def release_slot(self):
        """Release a slot after processing.

        Called by TQFullyAsyncRollouter in two cases:
        1. Normal path: after successfully writing sample to TQ (_process_single_sample_streaming)
        2. Error/drop path: when a sample fails BEFORE being written to TQ

        Notifies _slot_available so acquire_slot can wake up.
        """
        async with self._slot_available:
            self._pending_slots = max(0, self._pending_slots - 1)
            self._slot_available.notify_all()
        print(f"[ReplayBuffer][release_slot] pending={self._pending_slots}", flush=True)

    async def wait_and_sample(
        self,
        partition_id: str,
        batch_size: int,
    ) -> list[tuple[str, dict]] | None:
        """Block until enough finish samples are ready or production is fully complete.

        Called by TQFullyAsyncTrainer at the start of each training step.
        Returns keys whose metadata has current_status='finish'.

        Uses lock-free reads (safe due to Ray actor serialization + GIL for simple ops).
        Callers update metadata via update_metadata() and use Condition for safe writes.

        IMPORTANT — Termination semantics:
        -----------------------------------
        signal_finish() is called by Rollouter's _streaming_generation_main()
        only AFTER both:
          1. _feed_samples() has finished iterating the dataloader AND
          2. _processor_worker() has processed ALL samples from pending_queue
             (every sample has been inferred, written to TQ, and release_slot()'d)

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
        async with self._data_available:
            while True:
                # Check termination first
                if self._finished:
                    part = self.partitions.get(partition_id)
                    if part:
                        ready = [(k, v) for k, v in part.items() if v.get("current_status") == "finish"]
                        if ready:
                            print(
                                "[ReplayBuffer][wait_and_sample] Finished, returning "
                                f"{len(ready)} remaining samples (partial batch)",
                            )
                            return ready
                    print(
                        "[ReplayBuffer][wait_and_sample] Finished, no remaining samples, returning None",
                    )
                    return None

                # Check if enough finish samples are ready
                part = self.partitions.get(partition_id)
                if part is not None:
                    ready = [(k, v) for k, v in part.items() if v.get("current_status") == "finish"]

                    if len(ready) >= batch_size:
                        print(
                            f"[RB][wait_and_sample] Returning {len(ready[:batch_size])} samples from {partition_id}",
                        )
                        return ready[:batch_size]

                # Wait for _poll_from_tq to write new metadata or signal_finish
                await self._data_available.wait()

    async def reset_staleness(self) -> dict:
        """Reset the version window after parameter synchronization.

        Called by TQFullyAsyncTrainer (via Rollouter) after
        checkpoint_manager.update_weights(). This:

        1. Resets ``_version_slots = 0`` — unblocks acquire_slot() if it was
           blocked on the version window being full.

        Notifies _slot_available so acquire_slot can wake up.

        Returns:
            Dict with timing/metrics for logging.
        """
        # now = time.time()
        async with self._slot_available:
            prev_version_slots = self._version_slots

            # Core: reset version window to unblock acquire_slot()
            self._version_slots = 0

            # Timing metrics
            active_time = time.time()
            version_time = time.time()
            idle_ratio = 0

            timing_raw = {
                "fully_async/rollouter/active_time": active_time,
                "fully_async/rollouter/version_time": version_time,
                "fully_async/rollouter/idle_ratio": idle_ratio,
            }

            print(
                f"[ReplayBuffer][reset_staleness] "
                f"version_slots: {prev_version_slots} -> 0, "
                f"pending_slots: {self._pending_slots}, "
                f"idle_ratio: {idle_ratio:.4f}",
            )

            # Wake up acquire_slot waiters blocked on version window
            self._slot_available.notify_all()

        return timing_raw

    async def signal_finish(self):
        """Signal that production is fully complete — all samples are done.

        Called by TQFullyAsyncRollouter._streaming_generation_main() in its
        finally block, which runs ONLY after both:
          1. _feed_samples() has exhausted the dataloader (put None sentinel), AND
          2. _processor_worker() has drained pending_queue completely
             (all samples inferred, written to TQ, and release_slot()'d).

        This is the definitive "no more data ever" signal. When this fires,
        wait_and_sample() and acquire_slot() can safely return.
        There are guaranteed to be zero in-flight / un-finished samples.
        """
        # Wake up acquire_slot waiters (return False)
        async with self._slot_available:
            self._finished = True
            self._slot_available.notify_all()
        # Wake up wait_and_sample waiters (drain remaining)
        async with self._data_available:
            self._data_available.notify_all()

    # ======== Statistics ========

    def count_by_status(self, status: StatusType, partition_id: str | None = None) -> int:
        """Count samples by status. Lock-free read."""
        if partition_id:
            parts = [self.partitions.get(partition_id, {})]
        else:
            parts = list(self.partitions.values())
        return sum(1 for part in parts for v in part.values() if v.get("current_status") == status)

    def total_in_flight(self) -> int:
        """Total in-flight samples across all non-terminal states. Lock-free read."""
        return (
            self._pending_slots
            + self.count_by_status("pending")
            + self.count_by_status("running")
            + self.count_by_status("partial")
            + self.count_by_status("finish")
        )

    def get_statistics(self) -> dict:
        """Return statistics about the buffer state. Lock-free read."""
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
            "total_in_flight": self.total_in_flight(),
            # Layer 1: Physical slot control
            "pending_slots": self._pending_slots,
            "max_pending_slots": self.max_pending_slots,
            "available_physical_slots": max(0, self.max_pending_slots - self._pending_slots),
            # Layer 2: Version window control
            "version_slots": self._version_slots,
            "max_version_slots": self.max_version_slots,
            "available_version_slots": max(0, (self.max_version_slots or 0) - self._version_slots),
            # Control
            "finished": self._finished,
        }
