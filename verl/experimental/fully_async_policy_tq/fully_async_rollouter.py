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

"""TQFullyAsyncRollouter: Distributes prompts to TQ with slot-based backpressure control.

Key differences from FullyAsyncRollouter (MessageQueue-based):
- Does NOT call AgentLoopManager.generate_sequences() and wait for results
- Writes prompt data directly to TQ via kv_batch_put
- Uses ReplayBuffer.acquire_slot() for backpressure (instead of queue size checks)
- Worker side handles the actual generation asynchronously

Data Flow:
  dataloader -> prepare_single_generation_data -> acquire_slot -> kv_batch_put(prompt)
  -> [Worker picks up from RB] -> [Server generates] -> kv_batch_put(response) -> [Trainer consumes]
"""

import asyncio
import logging
import os
import time
from pprint import pformat

import ray
import torch

from verl.experimental.fully_async_policy.detach_utils import (
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.utils import need_reward_model
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq


@ray.remote(num_cpus=10, max_concurrency=100)
class TQFullyAsyncRollouter(SeparateRayPPOTrainer):
    """Async sample generator that writes prompts to TransferQueue.

    Responsibilities:
    1. Read data from dataloader
    2. Call prepare_single_generation_data to process data
    3. Acquire slot (blocking, for backpressure), write to TQ
    4. Manage checkpoint via CheckpointEngineManager
    """

    def __init__(
        self,
        config,
        tokenizer,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store tokenizer and processor
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine, "hybrid_engine is not supported in TQ mode"
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must >= 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, "trigger_parameter_sync_step must >= 1"

        self.use_reference_policy = False
        self.use_rm = need_reward_model(self.config)
        if self.use_rm:
            assert self.config.reward.reward_model.enable_resource_pool, (
                "GenRM/DisRM in fully async mode requires standalone mode (enable_resource_pool=True). "
                "Colocate mode is not supported because async rollout never pauses."
            )

        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== fully async TQ config ====================
        print("[TQFullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[TQFullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.replay_buffer_handle = None  # Set via set_replay_buffer()
        self.current_model_version = 0
        self.partition_id = config.trainer.get("partition_id", "train")

        # Reward loop manager
        self.reward_loop_manager = None

        # Elastic worker group (injected before init_workers)
        self._elastic_worker_group = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_pending_slots = None
        self.max_concurrent_samples = None

        # Statistics
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        self.global_steps = 1
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        # Concurrency control
        self.paused = False
        self.running = True

        # Active TQ write tasks
        self.active_tasks = set()

    def _init_async_objects(self):
        """Initialize asyncio synchronization primitives."""
        self.lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    async def set_replay_buffer(self, replay_buffer_handle):
        """Set the ReplayBuffer actor handle."""
        async with self.lock:
            self.replay_buffer_handle = replay_buffer_handle
            print("[TQFullyAsyncRollouter] ReplayBuffer handle set")

    async def set_max_required_samples(self):
        """Calculate and set max required samples based on config."""
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            # max_pending_slots controls how many samples can be in-flight in TQ
            self.max_pending_slots = self.max_required_samples
            # Also used as the ReplayBuffer's max_pending_slots
            self.max_concurrent_samples = min(
                self.max_required_samples,
                self.max_pending_slots,
            )

            print(
                f"[TQFullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_pending_slots: {self.max_pending_slots} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_total_train_steps(self):
        return self.total_train_steps

    def get_max_queue_size(self):
        """Return max pending slots (equivalent to max queue size in MQ version)."""
        return self.max_pending_slots

    def do_validate(self):
        """Run validation and return metrics."""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            # val_metrics: dict = self._validate()
            # TODO
            val_metrics = {}
            pass
        return timing_raw | val_metrics

    def _validate_config(self):
        """Validate asynchronous training configuration."""
        if not hasattr(self.config, "async_training"):
            raise ValueError("[TQFullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate_log_probs"

    async def init_workers(self):
        """Initialize distributed training workers."""
        self._init_async_objects()

    def _create_continuous_iterator(self):
        """Create a continuous data iterator across epochs."""
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _feed_and_write_loop(self):
        """Unified loop: read from dataloader, acquire slot, write to TQ.

        Merges the previous _feed_samples + _processor_worker + _process_single_sample_to_tq
        into a single task that handles the full pipeline: dataloader -> slot -> TQ.
        """
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Check pause condition (backpressure from RB)
            if self.paused or await self._should_pause_generation():
                print("[TQFullyAsyncRollouter][FeedLoop] Paused, waiting...")
                async with self.lock:
                    self.paused = True
                    self._resume_event.clear()

                resume_future = asyncio.ensure_future(self._resume_event.wait())
                try:
                    while self.active_tasks and not resume_future.done():
                        wait_set = set(self.active_tasks) | {resume_future}
                        done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {resume_future}
                        if actual_done:
                            async with self.lock:
                                for task in actual_done:
                                    self.active_tasks.discard(task)
                                    await task
                    if resume_future in done:
                        print("[TQFullyAsyncRollouter][FeedLoop] Resumed early")
                        break

                    if not resume_future.done():
                        self.idle_start_time = time.time()
                        await resume_future
                finally:
                    if not resume_future.done():
                        resume_future.cancel()
                        await asyncio.gather(resume_future, return_exceptions=True)
                continue  # Re-check pause after resume

            # Concurrency control: wait for a slot in active_tasks
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            # Prepare generation data
            full_batch = prepare_single_generation_data(batch_dict, self.config)
            sample_id = f"sample_{epoch}_{self.global_steps}"

            # Submit write task (acquire slot -> write to TQ)
            async with self.lock:
                task = safe_create_task(
                    self._write_single_to_tq(full_batch, sample_id),
                    name=sample_id,
                    task_set=self.active_tasks,
                )

            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[TQFullyAsyncRollouter][FeedLoop] "
                    f"Maximum steps reached: {self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End of dataloader: drain active tasks
        print("[TQFullyAsyncRollouter][FeedLoop] End of dataloader, draining active tasks...")
        while self.active_tasks:
            async with self.lock:
                if self.active_tasks:
                    done_tasks, self.active_tasks = await asyncio.wait(
                        self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done_tasks:
                        await task

        print(f"[TQFullyAsyncRollouter][FeedLoop] Completed. {self.global_steps} samples processed")

    async def _write_single_to_tq(self, full_batch, sample_id: str):
        """Acquire slot and write a single sample's prompt data to TQ.

        Args:
            full_batch: Prepared DataProto batch from prepare_single_generation_data.
            sample_id: Unique identifier for this sample.
        """
        try:
            for i in range(len(full_batch)):
                uid = f"uid_{sample_id}_{i}"

                # Extract non_tensor fields for metadata
                meta_fields = {}
                for nk in list(full_batch.non_tensor_batch.keys()):
                    val = full_batch.non_tensor_batch[nk]
                    if i < len(val):
                        meta_fields[nk] = val[i]

                # 1. Acquire slot (blocking, backpressure control)
                acquired = await self.replay_buffer_handle.acquire_slot.remote(timeout=None)
                if not acquired:
                    logging.warning("[TQFullyAsyncRollouter] Failed to acquire slot, stopping...")
                    return

                # 2. Build key
                key = f"{self.partition_id}_{uid}"

                # 3. Extract tensor fields for TQ
                fields = {}
                for bk in list(full_batch.batch.keys()):
                    tensor = full_batch.batch[bk]
                    if tensor.dim() >= 1 and i < tensor.shape[0]:
                        fields[bk] = tensor[i]

                # Add non-tensor metadata to fields
                for nk, nv in meta_fields.items():
                    fields[nk] = nv

                # 4. Determine prompt_len
                prompt_len = fields.get("input_ids", torch.tensor([])).shape[0]

                # 5. Write to TQ
                tags = {
                    "current_status": "pending",
                    "uid": uid,
                    "session_id": 0,
                    "trajectory_id": 0,
                    "start_model_version": self.current_model_version,
                    "end_model_version": self.current_model_version,
                    "prompt_len": prompt_len,
                    **meta_fields,
                }

                await tq.async_kv_batch_put(
                    keys=[key],
                    fields=fields,
                    tags=tags,
                    partition_id=self.partition_id,
                )

                self.total_generated_samples += 1

            self.processed_sample_count += 1

        except Exception as e:
            logging.exception(f"[TQFullyAsyncRollouter] Error writing sample {sample_id} to TQ: {e}")
            self.dropped_stale_samples += 1

    async def _streaming_generation_main(self):
        """Main entry point for streaming TQ-based generation."""
        print(
            f"[TQFullyAsyncRollouter] Starting TQ streaming mode, "
            f"max concurrent: {self.max_concurrent_samples}, "
            f"max pending slots: {self.max_pending_slots}"
        )

        # Single unified loop: dataloader -> acquire slot -> write to TQ
        self.feed_task = safe_create_task(self._feed_and_write_loop(), name="feed_and_write_loop")

        try:
            await self.feed_task
            print("[TQFullyAsyncRollouter] Feed and write loop completed")

        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Streaming error: {e}")
            raise e
        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)
            self.feed_task = None

            # Signal finish to ReplayBuffer
            if self.replay_buffer_handle is not None:
                await self.replay_buffer_handle.signal_finish.remote()

            async with self.lock:
                self.running = False

    async def fit(self):
        """Start the TQ-based async rollouter."""
        print("[TQFullyAsyncRollouter] Starting TQFullyAsyncRollouter...")

        if self.replay_buffer_handle is None:
            raise ValueError("ReplayBuffer not set. Call set_replay_buffer() first.")

        async with self.lock:
            self.paused = False
            self.running = True
            self._resume_event.set()

        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Error: {e}")
        finally:
            for t in [generation_task, monitor_task]:
                if t and not t.done():
                    t.cancel()
                    await asyncio.gather(t, return_exceptions=True)

        print("[TQFullyAsyncRollouter] fit completed")

    async def _async_monitor_loop(self):
        """Monitor loop for logging and recovery."""
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)

            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[TQFullyAsyncRollouter][Monitor] {pformat(stats)}")
                last_stats_time = current_time

            # Auto-resume if paused but should not be
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[TQFullyAsyncRollouter][Monitor] Auto-resuming")
                    self._resume_event.set()

    async def _should_pause_generation(self) -> bool:
        """Determine whether to pause generation based on buffer state."""
        if self.replay_buffer_handle is None:
            return False

        try:
            stats = await self.replay_buffer_handle.get_statistics.remote()
            pending = stats.get("pending_slots", 0)
        except Exception:
            return False

        if pending >= self.max_pending_slots:
            if not self.paused:
                print(f"[TQFullyAsyncRollouter][ShouldPause] pending_slots={pending} >= max={self.max_pending_slots}")
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[TQFullyAsyncRollouter][ShouldPause] "
                    f"staleness_samples {self.staleness_samples} >= max {self.max_required_samples}"
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        rb_stats = {}
        if self.replay_buffer_handle is not None:
            try:
                rb_stats = await self.replay_buffer_handle.get_statistics.remote()
            except Exception:
                pass

        stats = {
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/rb/pending_slots": rb_stats.get("pending_slots", 0),
            "monitor/rb/ready_count": rb_stats.get("total_ready", 0),
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_pending_slots": self.max_pending_slots,
            "static/current_model_version": self.current_model_version,
        }
        return stats

    # -------------------------------------------------------------------------
    # Elastic worker group injection (for trainer-side validation)
    # -------------------------------------------------------------------------

    def set_elastic_worker_group(self, worker_group: RayWorkerGroup):
        """Inject elastic worker group."""
        self._elastic_worker_group = worker_group

    def get_elastic_worker_group(self):
        """Return elastic worker group."""
        return self._elastic_worker_group
