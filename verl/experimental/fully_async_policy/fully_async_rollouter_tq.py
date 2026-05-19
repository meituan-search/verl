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
3. _should_pause_generation(): always returns False (slot-based flow control in _feed_samples
   replaces queue-size / staleness backpressure; _processor_worker is reused from base class)
6. reset_staleness() delegates to ReplayBuffer
"""

import asyncio
import logging
import os
import time

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

from verl.trainer.main_ppo_sync import list_of_dict_to_tensordict

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
        self.replay_buffer = None  # Ray Actor handle, set via set_replay_buffer()

        print("[TQFullyAsyncRollouter] initialized (TQ mode)")

    # ======== ReplayBuffer injection (replaces set_message_queue_client) ========

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle."""
        async with self.lock:
            self.replay_buffer = replay_buffer

            tq.init()
            print("[TQFullyAsyncRollouter] TQ initialized in Rollouter actor process", flush=True)

    # ======== Override: _feed_samples — add acquire_slot() ========

    async def _feed_samples(self):
        """Feed samples from dataloader to pending_queue, with source-level flow control.

        Key difference from base class: acquire_slot() is called BEFORE putting
        to pending_queue. This blocks the dataloader when too many samples are
        in-flight, replacing the need for _should_pause_generation().
        """
        print("[TQFullyAsyncRollouter][_feed_samples] STARTING", flush=True)
        continuous_iterator = self._create_continuous_iterator()
        feed_count = 0

        for epoch, batch_dict in continuous_iterator:
            print(f"[TQFullyAsyncRollouter][Feed] Acquiring slot for sample {feed_count}...", flush=True)
            acquired = await self.replay_buffer.acquire_slot.remote(timeout=None)
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
            print(
                f"[TQFullyAsyncRollouter][Feed] Put sample {feed_count - 1} to pending_queue (total={feed_count})",
                flush=True,
            )

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

    async def _should_pause_generation(self) -> bool:
        return False

    # ======== Override: _process_single_sample_streaming — MQ → TQ ========
    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample: generate via ALM, then write to TQ.

        Key difference from base class:
        - Instead of message_queue_client.put_sample(ray.cloudpickle.dumps(...)),
          writes data to TQ with status=finish tag.
        - RB's poll thread will detect finish and auto-release slot.
        """
        print(
            f"[TQFullyAsyncRollouter][_process_single] Starting generate for {rollout_sample.sample_id}...", flush=True
        )
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        print(
            f"[TQFullyAsyncRollouter][_process_single] generate done for {rollout_sample.sample_id}, writing to TQ...",
            flush=True,
        )
        rollout_sample.full_batch = ret
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        # Skip get_statistics() in TQ mode — it calls message_queue_client.get_statistics()
        # which is not available (replaced by ReplayBuffer). Use lightweight stats instead.
        # rollout_sample.rollout_status = await self.get_statistics()
        rollout_sample.rollout_status = {
            "monitor/active_tasks_size": len(self.active_tasks) if hasattr(self, "active_tasks") else 0,
            "count/total_generated_samples": self.total_generated_samples,
        }

        # ★ CORE CHANGE: Write to TQ instead of MessageQueue
        # Follow AgentLoopWorkerTQ._agent_loop_postprocess pattern:
        # - Use async_kv_batch_put (not blocking kv_batch_put)
        # - Split 2D [bsz, seq] batch into per-sample 1D dicts, then list_of_dict_to_tensordict
        #   This is critical because TQ's jagged layout requires each sample to be a 1-D tensor
        #   (matching AgentLoopWorkerTQ where output.as_dict() yields 1-D prompts/responses).
        base_key = rollout_sample.sample_id  # e.g. "sample_0_42"
        full_batch = rollout_sample.full_batch  # DataProto

        try:
            batch = full_batch.batch if hasattr(full_batch, "batch") else full_batch
            non_tensor = full_batch.non_tensor_batch if hasattr(full_batch, "non_tensor_batch") else {}

            # Determine batch size from tensor fields
            bsz = len(full_batch) if hasattr(full_batch, "__len__") else 1
            if "input_ids" in batch:
                bsz = batch["input_ids"].shape[0]
            elif "prompts" in batch:
                bsz = batch["prompts"].shape[0]

            # Tensor keys to extract per-sample (2D → 1D unbind along dim=0)
            per_sample_tensor_keys = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "response_mask",
                "prompts",
                "responses",
                "loss_mask",
            ]
            # Non-tensor / score keys to index per-sample
            per_sample_nt_keys = [
                "uid",
                "reward_score",
                "rm_scores",
                "rollout_log_probs",
                "old_log_probs",
                "ref_log_prob",
                "log_probs",
                "entropy",
            ]

            keys, fields, tags = [], [], []
            for idx in range(bsz):
                sample_key = f"{base_key}_{idx}" if bsz > 1 else base_key
                field = {}

                # Unbind 2D tensors along batch dimension → 1D per sample
                for tkey in per_sample_tensor_keys:
                    if tkey in batch:
                        field[tkey] = batch[tkey][idx]

                # Index non-tensor arrays along batch dimension
                for ntkey in per_sample_nt_keys:
                    if ntkey in non_tensor:
                        val = non_tensor[ntkey]
                        # Handle numpy arrays: index element-wise
                        if isinstance(val, np.ndarray):
                            field[ntkey] = val[idx]
                        elif isinstance(val, list) and len(val) > idx:
                            field[ntkey] = val[idx]
                        else:
                            field[ntkey] = val
                    elif ntkey in batch:
                        field[ntkey] = batch[ntkey][idx]

                # Per-sample metadata: model version from generate result (ret)
                # min_global_steps / max_global_steps are set by FullyAsyncLLMServerClient.generate()
                # into TokenOutput.extra_fields, then collected by AgentLoop into non_tensor_batch.
                field["global_steps"] = self.global_steps
                field["start_model_version"] = non_tensor.get("min_global_steps", idx)[idx]
                field["end_model_version"] = non_tensor.get("max_global_steps", idx)[idx]

                # Ensure loss_mask exists (required by _update_actor pipeline).
                # In AgentLoopWorkerTQ._agent_loop_postprocess: field["loss_mask"] = field["response_mask"]
                # The rollout output batch typically does NOT contain loss_mask, so we derive it from response_mask.
                # Use .clone() to avoid potential shared-memory issues with jagged/nested tensors.
                if "response_mask" in field:
                    if "loss_mask" not in field:
                        field["loss_mask"] = (
                            field["response_mask"].clone()
                            if hasattr(field["response_mask"], "clone")
                            else field["response_mask"]
                        )
                    lm_shape = field["loss_mask"].shape
                    print(
                        f"[RollouterTQ] {sample_key}: loss_mask set, shape={lm_shape}",
                        flush=True,
                    )
                else:
                    print(
                        f"[TQFullyAsyncRollouter][_process_single] WARNING: {sample_key} no response_mask/loss_mask!",
                        flush=True,
                    )

                # Compute seq_len from 1-D prompts/responses (same as AgentLoopWorkerTQ)
                prompts_1d = field.get("prompts")
                responses_1d = field.get("responses")
                prompt_len = prompts_1d.size(0) if (prompts_1d is not None and hasattr(prompts_1d, "size")) else 1
                response_len = (
                    responses_1d.size(0) if (responses_1d is not None and hasattr(responses_1d, "size")) else 0
                )
                seq_len = prompt_len + response_len

                keys.append(sample_key)
                fields.append(field)
                tags.append(
                    {
                        "current_status": "finish",
                        "uid": sample_key,
                        "start_model_version": field["start_model_version"],
                        "end_model_version": field["end_model_version"],
                        "prompt_len": int(prompt_len),
                        "response_len": int(response_len),
                        "seq_len": int(seq_len),
                    }
                )

            # Convert list of per-sample dicts → TensorDict (handles jagged correctly)
            print(
                f"[TQFullyAsyncRollouter][_process_single] Preparing TQ write for {base_key}, bsz={bsz}",
                flush=True,
            )
            td_fields = list_of_dict_to_tensordict(fields)
            td_keys_str = list(td_fields.keys())
            print(
                f"[RollouterTQ] TensorDict ready for {base_key}, keys={td_keys_str}, bsz={td_fields.batch_size}",
                flush=True,
            )

            t_write_start = time.time()
            await tq.async_kv_batch_put(
                keys=keys,
                fields=td_fields,
                tags=tags,
                partition_id="train",
            )
            t_write_end = time.time()
            self.total_generated_samples += bsz
            keys_str = ", ".join(keys[:3]) + ("..." if len(keys) > 3 else "")
            wt = t_write_end - t_write_start
            total_gen = self.total_generated_samples
            print(
                f"[RollouterTQ] Wrote {bsz} samples [{keys_str}] to TQ ({total_gen} total, {wt:.2f}s)",
                flush=True,
            )

            for sample_key, tag in zip(keys, tags, strict=False):
                finish_meta = dict(tag)
                await asyncio.wrap_future(
                    self.replay_buffer.mark_finish.remote(
                        key=sample_key, partition_id="train", meta=finish_meta
                    ).future()
                )
        except Exception as e:
            logger.exception(f"[TQFullyAsyncRollouter] Failed to write {base_key} to TQ: {e}")
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
        print("[TQFullyAsyncRollouter][_streaming_generation_main] ENTER", flush=True)
        if self.async_rollout_manager is None:
            print(
                "[TQFullyAsyncRollouter][_streaming_generation_main] Calling _init_async_rollout_manager...", flush=True
            )
            await self._init_async_rollout_manager()
            print("[TQFullyAsyncRollouter][_streaming_generation_main] _init_async_rollout_manager DONE", flush=True)
        else:
            print(
                "[TQFullyAsyncRollouter][_streaming_generation_main] "
                "async_rollout_manager already exists, skipping init",
                flush=True,
            )

        print(
            f"[TQFullyAsyncRollouter] Start streaming mode, max concurrent: {self.max_concurrent_samples}", flush=True
        )

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

    # ======== Override: reset_staleness — resume + delegate to RB ========

    async def reset_staleness(self):
        """Reset version window after parameter update."""
        # Reset RB version window (no longer passes active_task_count —
        # RB's staleness control is purely via acquire_slot dual-condition)
        print("[RollouterTQ][reset_staleness] calling RB.reset_staleness (version window reset)", flush=True)
        rb_timing = await self.replay_buffer.reset_staleness.remote()
        print(f"[TQFullyAsyncRollouter][reset_staleness] RB.reset_staleness DONE {rb_timing}", flush=True)
        return rb_timing

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
            # Static config
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_concurrent_samples": self.max_concurrent_samples,
            # RB stats (if available)
            **rb_stats,
        }

        return stats
