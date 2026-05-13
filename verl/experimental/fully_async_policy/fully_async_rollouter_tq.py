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

"""TQFullyAsyncRollouter: FullyAsyncRollouter with TQ + ReplayBuffer.

Key changes from base class (FullyAsyncRollouter):
1. _feed_samples(): acquire_slot() before putting to pending_queue (source-level flow control)
2. _process_single_sample_streaming(): write to TQ instead of MessageQueue
3. _processor_worker(): remove _should_pause_generation() check (slot handles backpressure)
4. Remove: _should_pause_generation(), max_queue_size, staleness_samples manual counting
5. set_message_queue_client() → set_replay_buffer()
6. reset_staleness() delegates to ReplayBuffer
7. finish signal via ReplayBuffer.signal_finish() instead of MessageQueue.put_sample(None)
"""

import asyncio
import logging
import os
import time
from pprint import pformat

import numpy as np
import ray

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter,
)

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TQFullyAsyncRollouter(FullyAsyncRollouter):
    """
    FullyAsyncRollouter variant that uses TransferQueue + ReplayBuffer instead of MessageQueue.

    Core design:
    - ReplayBuffer.acquire_slot() in _feed_samples() provides source-level flow control
      (replaces _should_pause_generation + MessageQueue.queue_size backpressure)
    - Generated samples are written to TQ (zero-copy) instead of MessageQueue (pickle)
    - FullyAsyncAgentLoopManager is unchanged — still used for actual generation
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
    ):
        # Call parent __init__ (sets up datasets, dataloader, basic config)
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, device_name=device_name)

        # ==================== TQ-specific overrides ====================
        # Replace MessageQueue client with ReplayBuffer handle
        self.replay_buffer = None  # Ray Actor handle, set via set_replay_buffer()

        # Remove MessageQueue-related fields (replaced by RB slot control)
        # - self.message_queue_client: replaced by self.replay_buffer
        # - self.max_queue_size: replaced by RB.max_pending_slots
        # - self.staleness_samples: managed by RB.reset_staleness()

        # Track current param version for TQ tags
        self.current_param_version = 0

        print("[TQFullyAsyncRollouter] initialized (TQ mode)")

    # ======== ReplayBuffer injection (replaces set_message_queue_client) ========

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle.

        Replaces set_message_queue_client() from base class.
        """
        async with self.lock:
            self.replay_buffer = replay_buffer
            # Also store in message_queue_client for base class compatibility
            # (some base class methods may check for its existence)
            self.message_queue_client = None  # Explicitly mark as unused
        print("[TQFullyAsyncRollouter] ReplayBuffer handle set")

    def get_max_pending_slots(self):
        """Return max_pending_slots for main.py to create RB with correct size."""
        return getattr(self, "max_required_samples", 256)

    # ======== Override: _feed_samples — add acquire_slot() ========

    async def _feed_samples(self):
        """Feed samples from dataloader to pending_queue, with source-level flow control.

        Key difference from base class: acquire_slot() is called BEFORE putting
        to pending_queue. This blocks the dataloader when too many samples are
        in-flight, replacing the need for _should_pause_generation().
        """
        continuous_iterator = self._create_continuous_iterator()
        feed_count = 0

        for epoch, batch_dict in continuous_iterator:
            # ★ CORE CHANGE: acquire slot at dataloader source (blocking)
            # This replaces _should_pause_generation() backpressure.
            # When RB.pending_slots >= max_pending_slots, this blocks.
            if self.replay_buffer is not None:
                acquired = await asyncio.wrap_future(self.replay_buffer.acquire_slot.remote(timeout=None).future())
                if not acquired:
                    print(
                        f"[TQFullyAsyncRollouter][Feed] ReplayBuffer finished or closed, "
                        f"stop feeding after {feed_count} samples"
                    )
                    break

            # Prepare data (same as base class)
            full_batch = prepare_single_generation_data(batch_dict, self.config)
            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)
            feed_count += 1

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[TQFullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[TQFullyAsyncRollouter][Feed] Sample addition is complete, {feed_count} samples have been added")

        # Notify ReplayBuffer that production is done
        if self.replay_buffer is not None:
            ray.get(self.replay_buffer.signal_finish.remote())

    # ======== Override: _processor_worker — remove _should_pause_generation ========

    async def _processor_worker(self):
        """Streaming worker coroutine.

        Key difference from base class:
        - REMOVED: `await self._should_pause_generation()` check
          (slot-based flow control in _feed_samples makes this unnecessary)
        - KEPT: paused drain logic (still needed for parameter sync)
        - REMOVED: `self.staleness_samples += 1` (managed by RB)
        """
        while True:
            # ★ CHANGED: only check paused (for param sync drain), NOT _should_pause_generation
            if self.paused:
                print(
                    "[TQFullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                    self._resume_event.clear()

                resume_future = asyncio.ensure_future(self._resume_event.wait())
                try:
                    # Drain: wait for active tasks or resume signal (same as base class)
                    while self.active_tasks and not resume_future.done():
                        wait_set = set(self.active_tasks) | {resume_future}
                        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {resume_future}
                        if actual_done:
                            async with self.lock:
                                for task in actual_done:
                                    self.active_tasks.discard(task)
                                    await task
                        if resume_future in done:
                            print(
                                "[TQFullyAsyncRollouter][Processor] "
                                "Drain interrupted by resume signal, resuming generation early "
                                f"(active tasks remaining: {len(self.active_tasks)})"
                            )
                            break

                    # block until resuming
                    if not resume_future.done():
                        self.idle_start_time = time.time()
                        await resume_future
                finally:
                    if not resume_future.done():
                        resume_future.cancel()
                        await asyncio.gather(resume_future, return_exceptions=True)
                continue

            # Get sample from pending_queue
            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()

            # ★ REMOVED: self.staleness_samples += 1 (RB manages this now)

            if rollout_sample is None:
                print(
                    "[TQFullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                break

            # GPU concurrency limit (kept from base class)
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            # Submit single sample processing
            if self.paused:
                await self._resume_event.wait()
            async with self.lock:
                task = safe_create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )

    # ======== Override: _process_single_sample_streaming — MQ → TQ ========

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample: generate via ALM, then write to TQ.

        Key difference from base class:
        - Instead of message_queue_client.put_sample(ray.cloudpickle.dumps(...)),
          writes data to TQ with status=finish tag.
        - RB's poll thread will detect finish and auto-release slot.
        """
        # Generate via AgentLoopManager (UNCHANGED from base class)
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        rollout_sample.full_batch = ret
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()

        # ★ CORE CHANGE: Write to TQ instead of MessageQueue
        key = rollout_sample.sample_id  # e.g. "sample_0_42"

        try:
            tq.kv_batch_put(
                keys=[key],
                partition_id="train",
                fields=[rollout_sample],  # Full RolloutSample as field data
                tags=[
                    {
                        "current_status": "finish",
                        "uid": key,
                        "start_model_version": self.current_param_version,
                        "end_model_version": self.current_param_version,
                        "prompt_len": 0,  # TODO: extract from ret if needed
                        "response_len": 0,  # TODO: extract from ret if needed
                    }
                ],
            )
            self.total_generated_samples += 1
        except Exception as e:
            logger.error(f"[TQFullyAsyncRollouter] Failed to write {key} to TQ: {e}")
            # Release slot on error so we don't leak
            if self.replay_buffer is not None:
                ray.get(self.replay_buffer.release_slot.remote())

        self.processed_sample_count += 1

    # ======== Override: _streaming_generation_main — finish signal via RB ========

    async def _streaming_generation_main(self):
        """Main entry for stream processing.

        Key difference from base class:
        - Finish signal goes through RB.signal_finish() instead of MQ.put_sample(None)
        """
        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        print(f"[TQFullyAsyncRollouter] Start streaming mode, max concurrent: {self.max_concurrent_samples}")

        # Start feed and processor tasks (same as base class)
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[TQFullyAsyncRollouter] Sample feed completed")
            await self.processor_task
            print("[TQFullyAsyncRollouter] Streaming process completed")
            await self.pending_queue.join()
            print("[TQFullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Streaming process exception: {e}")
            raise e
        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # ★ CHANGED: Signal finish via RB (already done in _feed_samples, but ensure here too)
            # Note: _feed_samples already calls signal_finish when it ends.
            # This is a safety net for error paths.
            if self.replay_buffer is not None:
                try:
                    ray.get(self.replay_buffer.signal_finish.remote())
                except Exception:
                    pass

            async with self.lock:
                self.running = False

    # ======== Override: fit — check replay_buffer instead of message_queue_client ========

    async def fit(self):
        """Start the async rollouter."""
        print("[TQFullyAsyncRollouter] Starting TQFullyAsyncRollouter...")

        if self.replay_buffer is None:
            raise ValueError("ReplayBuffer not set. Call set_replay_buffer() first.")

        # Set running status (same as base class)
        async with self.lock:
            self.paused = False
            self.running = True
            self._resume_event.set()

        # Create main tasks (same structure as base class)
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[TQFullyAsyncRollouter] Rollouter fit completed")

    # ======== Override: reset_staleness — delegate to RB ========

    async def reset_staleness(self):
        """Reset staleness after parameter update.

        Delegates to ReplayBuffer.reset_staleness() which manages version tracking
        centrally. Also wakes up the processor if paused.
        """
        async with self.lock:
            self.paused = False
            self._resume_event.set()

        # Delegate staleness tracking to RB
        if self.replay_buffer is not None:
            active_task_count = len(self.active_tasks)
            timing_raw = ray.get(self.replay_buffer.reset_staleness.remote(active_task_count=active_task_count))
        else:
            # Fallback if RB not set (shouldn't happen in normal operation)
            timing_raw = {}
            rollout_version_time = max(time.time() - self.step_start_time, 1e-6)
            if self.idle_start_time > self.step_start_time:
                rollout_active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
            else:
                rollout_active_time = rollout_version_time
                idle_ratio = 0
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            self.current_param_version = getattr(self, "current_param_version", 0) + 1

            print(
                f"[TQFullyAsyncRollouter][reset_staleness] "
                f"model_version={self.current_param_version}, "
                f"idle_ratio={idle_ratio:.4f}"
            )

        self.step_start_time = time.time()

        return timing_raw

    # ======== Override: _async_monitor_loop — remove _should_pause_generation ========

    async def _async_monitor_loop(self):
        """Monitor loop for logging and recovery.

        Key difference from base class:
        - REMOVED: `_should_pause_generation()` check for auto-resume
          (slot-based flow control doesn't use pause/resume for backpressure)
        - KEPT: basic statistics logging
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)

            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[TQFullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # ★ REMOVED: auto-resume based on _should_pause_generation()
            # With slot-based flow control, we don't pause based on queue size.
            # The paused state is only used for param-sync drain, which is handled
            # by reset_staleness() setting _resume_event.

    # ======== REMOVED: _should_pause_generation ========
    # No longer needed — acquire_slot() in _feed_samples provides all necessary backpressure.

    # ======== Override: get_statistics — source from RB instead of MQ ========

    async def get_statistics(self) -> dict:
        """Gather statistics from RB and local state."""
        rb_stats = {}
        if self.replay_buffer is not None:
            rb_stats = ray.get(self.replay_buffer.get_statistics.remote())

        stats = {
            # Monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            # Counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/processed_sample_count": self.processed_sample_count,
            "count/current_param_version": self.current_param_version,
            # Static config
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_concurrent_samples": self.max_concurrent_samples,
            # RB stats (if available)
            **rb_stats,
        }

        return stats

    # ======== Override: set_max_required_samples — also init TQ ========

    async def set_max_required_samples(self):
        """Set max required samples and initialize TQ."""
        await super().set_max_required_samples()

        # Initialize TQ on this worker
        try:
            tq.init()
            print("[TQFullyAsyncRollouter] TQ initialized")
        except Exception as e:
            print(f"[TQFullyAsyncRollouter] TQ init warning (may already be initialized): {e}")

        print(
            f"[TQFullyAsyncRollouter] required_samples : {self.required_samples} "
            f"max_required_samples: {self.max_required_samples} "
            f"total_train_steps: {self.total_train_steps} "
            f"total_rollout_steps: {self.total_rollout_steps} "
            f"max_concurrent_samples: {self.max_concurrent_samples} "
        )
