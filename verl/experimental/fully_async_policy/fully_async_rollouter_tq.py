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

import asyncio
import logging
import os

import numpy as np
import ray

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.trainer.main_ppo_sync import AgentLoopWorkerTQ
from verl.utils import tensordict_utils as tu

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncAgentLoopManagerTQ(FullyAsyncAgentLoopManager):
    """Agent loop manager that uses :class:`AgentLoopWorkerTQ` workers.

    Key difference from base :class:`FullyAsyncAgentLoopManager`:
    - Overrides ``generate_sequences_single`` to convert ``DataProto`` → ``TensorDict``
      before dispatching to workers, aligning with :class:`AgentLoopWorkerTQ`'s
      ``generate_sequences`` signature which expects ``TensorDict`` (not ``DataProto``).
    - Default ``wait=True``: waits for all tasks to complete before returning.
      This ensures Rollouter knows exactly when generation finishes (avoids deadlock).
    """

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(AgentLoopWorkerTQ)
        super().__init__(*args, **kwargs)

    async def generate_sequences_single(self, prompts):
        """Convert DataProto to TensorDict, then dispatch to agent loop worker.

        Aligns with main_ppo_sync.py PPOTrainer.step() which calls::

            batch = tu.get_tensordict(batch_dict)
            self.async_rollout_manager.generate_sequences(batch)

        Args:
            prompts: Input batch (DataProto or TensorDict).

        Returns:
            None — data is written directly to TransferQueue by the worker.
        """
        # Convert DataProto → TensorDict (align with main_ppo_sync.py input side)
        # Manually merge batch + non_tensor_batch into a TensorDict, and lift meta_info
        # fields (validate, global_steps) into the TensorDict as NonTensorData — this
        # avoids the key collision in DataProto.to_tensordict() which also merges meta_info.
        if isinstance(prompts, DataProto):
            from verl.utils import tensordict_utils as tu

            tensor_batch = prompts.batch.to_dict() if prompts.batch is not None else {}
            for key, val in prompts.non_tensor_batch.items():
                tensor_batch[key] = val
            # Lift critical meta_info fields that generate_sequences expects as batch keys
            if "validate" in prompts.meta_info:
                tensor_batch["validate"] = prompts.meta_info["validate"]
            if "global_steps" in prompts.meta_info:
                tensor_batch["global_steps"] = prompts.meta_info["global_steps"]
            prompts = tu.get_tensordict(tensor_batch)

        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(prompts, wait=True)
        return await asyncio.wrap_future(output_future.future())


class FullyAsyncRollouterTQ(FullyAsyncRollouter):
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
        self.replay_buffer = None  # Ray Actor handle, set via set_replay_buffer()

        self.agent_loop_manager_class = FullyAsyncAgentLoopManagerTQ

        print("[TQFullyAsyncRollouter] initialized (TQ mode)")

    # ======== ReplayBuffer injection (replaces set_message_queue_client) ========

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle."""
        async with self.lock:
            self.replay_buffer = replay_buffer

            tq.init()
            print("[TQFullyAsyncRollouter] TQ initialized in Rollouter actor process", flush=True)

    async def _feed_samples(self):
        """Feed samples from dataloader to pending_queue, with source-level flow control.

        Key difference from base class: acquire_slot() is called BEFORE putting
        to pending_queue. This blocks the dataloader when too many samples are
        in-flight, replacing the need for _should_pause_generation().

        Alignment with main_ppo_sync.py PPOTrainer.step():1740-1758:
            - Do NOT repeat(n) here — let AgentLoopWorkerTQ._run_prompt loop n times
              via __rollout_n__ field (avoids double-repeat bug).
            - Set uid/__rollout_n__/sample_id/global_steps in batch_dict (plain dict)
              BEFORE calling tu.get_tensordict(), so these np.array values become NonTensorStack
              (supporting per-index access in AgentLoopWorkerTQ.generate_sequences).
        """
        print("[TQFullyAsyncRollouter][_feed_samples] STARTING", flush=True)
        continuous_iterator = self._create_continuous_iterator()
        rollout_n = self.config.actor_rollout_ref.rollout.n

        for epoch, batch_dict in continuous_iterator:
            sample_id = f"sample_{epoch}_{self.global_steps}"
            acquired = await self.replay_buffer.acquire_slot.remote(timeout=None, uid=sample_id)
            if not acquired:
                print(
                    f"[TQFullyAsyncRollouter][Feed] ReplayBuffer finished or closed, "
                    f"stop feeding after {self.global_steps} samples"
                )
                break

            # Inject fields into batch_dict (plain dict) BEFORE tu.get_tensordict().
            # All np.array values become NonTensorStack via get_tensordict:424,
            # supporting per-index access in AgentLoopWorkerTQ.generate_sequences().
            batch_dict["uid"] = np.array([sample_id], dtype=object)
            batch_dict["__rollout_n__"] = np.full(1, rollout_n, dtype=np.int64)
            batch_dict["sample_id"] = np.array([sample_id], dtype=object)
            batch_dict["global_steps"] = np.full(1, self.global_steps, dtype=np.int64)

            # Convert to TensorDict (np.array values → NonTensorStack via get_tensordict:424)
            full_batch = tu.get_tensordict(batch_dict)

            # Set agent_name for non-multi-turn mode (same as prepare_single_generation_data)
            if not self.config.actor_rollout_ref.rollout.multi_turn.enable:
                batch_dict["agent_name"] = np.array(["single_turn_agent"], dtype=object)
                full_batch = tu.get_tensordict(batch_dict)

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

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
        print(f"[TQFullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _should_pause_generation(self) -> bool:
        return False

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample: generate via ALM worker (which writes to TQ blocking), then release slot.

        Simplified from base class:
        - Base class: generate → put to MessageQueue
        - TQ path: generate + TQ write both happen INSIDE AgentLoopWorkerTQ
          (via overridden _agent_loop_postprocess)
          → we just call it and release the slot

        Note: full_batch now has bsz=1 (no repeat(n)), and contains __rollout_n__.
              AgentLoopWorkerTQ._run_prompt will loop n times internally to produce
              n responses per prompt, each written to TQ with a unique key.
        """
        try:
            await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
            self.total_generated_samples += 1
        except Exception as e:
            logger.exception(f"[TQFullyAsyncRollouter] Failed to process {rollout_sample.sample_id}: {e}")
        finally:
            # Always release the slot regardless of success/failure
            await self.replay_buffer.release_slot.remote()

        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        """Main entry for stream processing."""
        # Start feed and processor tasks (same as base class)
        print("[TQFullyAsyncRollouter] Starting feed_task...", flush=True)
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        print("[TQFullyAsyncRollouter] Starting processor_task...", flush=True)
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

            await self.replay_buffer.signal_finish.remote()

            async with self.lock:
                self.running = False

    async def reset_staleness(self):
        """Reset version window after parameter update."""
        rb_timing = await self.replay_buffer.reset_staleness.remote()
        return rb_timing

    async def get_statistics(self) -> dict:
        """Gather statistics from RB and local state."""
        rb_stats = {}
        if self.replay_buffer is not None:
            rb_stats = await self.replay_buffer.get_statistics.remote()

        stats = {
            # Monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            # Counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/processed_sample_count": self.processed_sample_count,
            # Static config
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_concurrent_samples": self.max_concurrent_samples,
            # RB stats (if available)
            **rb_stats,
        }

        return stats
