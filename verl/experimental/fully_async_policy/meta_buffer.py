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

"""MetaBuffer: lightweight metadata channel replacing MessageQueue + ReplayBuffer.

In fully_async separated architecture:
- Tensor data flows through TransferQueue (zero-copy)
- Metadata (key, status, seq_len, etc.) is synchronized via TQ kv_list
- This buffer caches metadata and provides: wait-and-sample, backpressure, finish signal

Slot control for TQ request count:
- acquire_slot(): Rollouter calls before writing to TQ. Blocks if too many pending.
- release_slot(): Called automatically when status transitions to 'success' (via kv_list poll).
- This ensures TQ doesn't accumulate unbounded requests.

Difference from ReplayBuffer (main_ppo_sync):
- No global_steps synchronization (async producer-consumer)
- sample() supports waiting + partial return when finished
"""

import logging
import os
import threading
import time
from collections import defaultdict

import ray

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote(num_cpus=1)
class MetaBuffer:
    """Lightweight metadata channel backed by TQ kv_list polling.

    Replaces both MessageQueue (signal channel) and ReplayBuffer (meta storage)
    in the fully_async policy training pipeline.

    Provides slot-based backpressure: producers must acquire a slot before writing
    to TQ, ensuring bounded in-flight requests.
    """

    def __init__(self, max_pending_slots: int = 256, poll_interval: float = 1.0):
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.lock = threading.Lock()
        self.poll_interval = poll_interval
        self._finished = False

        # Slot control for TQ request count
        self.max_pending_slots = max_pending_slots
        self._pending_slots = 0  # acquired but not yet completed (status=running)
        self._slot_available = threading.Condition(self.lock)

        # Background thread: sync meta from TQ kv_list
        self._poll_thread = threading.Thread(target=self._poll_from_tq, daemon=True)
        self._poll_thread.start()

    def _poll_from_tq(self):
        """Background thread that polls TQ for metadata updates.

        Also auto-releases slots when entries transition from 'running' to 'success'.
        """
        try:
            while True:
                data = tq.kv_list()
                if data is not None:
                    for partition_id, items in data.items():
                        with self.lock:
                            prev_keys = set(self.partitions.get(partition_id, {}).keys())
                            self.partitions[partition_id].update(items)
                            # Detect status transitions: running -> success => release slot
                            for key, meta in items.items():
                                if meta.get("status") == "success" and key in prev_keys:
                                    prev_meta = self.partitions[partition_id].get(key, {})
                                    if prev_meta.get("status") == "running":
                                        self._pending_slots -= 1
                                        self._slot_available.notify()
                                        logger.debug(f"[MetaBuffer] Slot released for {key} (running->success)")
                time.sleep(self.poll_interval)
        except Exception as e:
            logger.error(f"MetaBuffer _poll_from_tq error: {e}")
            os._exit(1)

    # ======== Slot control (backpressure for TQ request count) ========

    def acquire_slot(self, timeout: float | None = None) -> bool:
        """Acquire a slot before writing to TQ.

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
        """Manually release a slot (e.g., on error/drop)."""
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

    # ======== Consumer interface (Trainer) ========

    def wait_and_sample(
        self,
        partition_id: str,
        batch_size: int,
    ) -> list[tuple[str, dict]] | None:
        """Block until enough samples are ready or finish signal received.

        Args:
            partition_id: Partition to sample from, e.g. "train" or "val".
            batch_size: Desired number of samples.

        Returns:
            List of (key, tag) tuples. Length may be < batch_size when finished.
            Returns None if finished and no samples available.
        """
        while True:
            with self.lock:
                part = self.partitions.get(partition_id)
                if part is None:
                    # Partition not yet created (no samples written yet), wait
                    pass
                else:
                    ready = [(k, v) for k, v in part.items() if v.get("status") == "success"]

                    if len(ready) >= batch_size:
                        return ready[:batch_size]

                    if self._finished:
                        return ready if ready else None

            time.sleep(self.poll_interval)

    def remove(self, partition_id: str, keys: list[str]):
        """Remove sampled metadata after training consumption."""
        with self.lock:
            part = self.partitions.get(partition_id, {})
            for k in keys:
                part.pop(k, None)

    # ======== Dispatch interface (Rollouter dispatch loop) ========

    def get_running_keys(self, partition_id: str | None = None, limit: int = 0) -> list[str]:
        """Get keys with status='running' that are ready for dispatch to AgentLoopWorker.

        Args:
            partition_id: Filter by partition. None = all partitions.
            limit: Max number of keys to return. 0 = all.

        Returns:
            List of keys that have status='running' (written to TQ but not yet processed).
        """
        with self.lock:
            result = []
            pids = [partition_id] if partition_id else list(self.partitions.keys())
            for pid in pids:
                for k, v in self.partitions.get(pid, {}).items():
                    if v.get("status") == "running":
                        result.append(k)
                        if limit > 0 and len(result) >= limit:
                            break
                if limit > 0 and len(result) >= limit:
                    break
            return result

    # ======== Producer / control signals ========

    def pending_count(self) -> int:
        """Count of samples with status='running' across all partitions."""
        with self.lock:
            return sum(1 for part in self.partitions.values() for v in part.values() if v.get("status") == "running")

    def ready_count(self) -> int:
        """Count of samples with status='success' available for consumption."""
        with self.lock:
            return sum(1 for part in self.partitions.values() for v in part.values() if v.get("status") == "success")

    def signal_finish(self):
        """Signal that production is complete (no more samples will arrive)."""
        with self._slot_available:
            self._finished = True
            self._slot_available.notify_all()

    def is_finished(self) -> bool:
        """True if finish signaled and no pending slots remain."""
        with self.lock:
            return self._finished and self._pending_slots == 0

    def get_statistics(self) -> dict:
        """Return statistics about the buffer state."""
        with self.lock:
            partition_stats = {}
            for pid, part in self.partitions.items():
                running = sum(1 for v in part.values() if v.get("status") == "running")
                success = sum(1 for v in part.values() if v.get("status") == "success")
                partition_stats[pid] = {"running": running, "success": success, "total": len(part)}
            return {
                "partitions": partition_stats,
                "total_pending": self.pending_count,
                "total_ready": self.ready_count,
                "pending_slots": self._pending_slots,
                "max_pending_slots": self.max_pending_slots,
                "available_slots": max(0, self.max_pending_slots - self._pending_slots),
                "finished": self._finished,
            }
