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
import time
from collections import defaultdict
from pprint import pformat

import numpy as np
import ray
from ray import ObjectRef

from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    merge_rollout_sample,
    prepare_single_generation_data,
    assemble_batch_from_rollout_samples,
)
from recipe.fully_async_policy.message_queue import MessageQueueClient
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.ray_trainer import process_validation_metrics
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.consumer_task_validate = None
        self.processor_task_validate = None
        self.feed_task_validate = None
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.data.val_batch_size == 1, "val_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.global_steps = 0
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Initialize async locks directly
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.result_queue = asyncio.Queue()
        self.cancel_queue = asyncio.Queue()

        self.validate_task = None
        self.pending_queue_validate = asyncio.Queue(maxsize=128)
        self.active_tasks_validate = set()
        self.result_queue_validate = asyncio.Queue()
        self.cancel_queue_validate = asyncio.Queue()
        self.total_generated_samples_validate = 0
        self.global_steps_validate = 0

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
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

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = (
                len(self.active_tasks)
                + self.result_queue.qsize()
                + self.cancel_queue.qsize()
                + await self.message_queue_client.get_queue_size()
            )
            timing_raw = {}
            idle_ratio = None
            if self.idle_start_time is not None and self.version_start_time is not None:
                rollout_active_time = self.idle_start_time - self.version_start_time
                rollout_version_time = time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            data = ValidateMetrics(timing_raw=timing_raw, metrics=None, param_version=version)
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))
            self.version_start_time = time.time()

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    def _task_exception_handler(self, task: asyncio.Task):
        """Handle task exceptions and log them"""
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Task was cancelled, this is expected
        except Exception as e:
            print(f"[FullyAsyncRollouter] Task {task.get_name()} failed with exception: {e}")
            raise e

    async def safe_create_task(self, coro, name: str, task_set: set = None):
        """Safely create a task with exception handling

        Args:
            coro: The coroutine to run
            name: Name for the task
            task_set: Optional set to add the task to

        Returns:
            The created asyncio.Task
        """
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(self._task_exception_handler)
        if task_set is not None:
            task_set.add(task)
        return task

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(batch_dict, self.config.actor_rollout_ref.rollout.n)

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples"
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                while self.active_tasks:
                    async with self.lock:
                        # After acquiring the lock, the number of active_tasks may change, need to be verified again
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task

                async with self.lock:
                    while self.paused:
                        self.idle_start_time = time.time()
                        await self.condition.wait()
                continue

            simple_from_cancel_queue = False
            if not self.cancel_queue.empty():
                rollout_sample = await self.cancel_queue.get()
                simple_from_cancel_queue = True
            else:
                rollout_sample = await self.pending_queue.get()
                self.staleness_samples += 1

            if rollout_sample == "DONE":
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
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

            # Check whether the number of concurrent tasks exceeds the limit
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                    for task in done_tasks:
                        await task

            # Submit single sample processing
            async with self.lock:
                # After the pause is over, the lock is acquired and it is necessary
                # to determine whether it is the pause phase, otherwise continue to wait
                while self.paused:
                    await self.condition.wait()
                await self.safe_create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )

            if simple_from_cancel_queue:
                self.cancel_queue.task_done()
            else:
                self.pending_queue.task_done()

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        agent_loop_output_list = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        rollout_sample.agent_loop_output_list = agent_loop_output_list

        is_cancel = False
        for agent_loop in agent_loop_output_list:
            if not is_cancel and agent_loop.is_cancel:
                is_cancel = True

        if is_cancel:
            # Put in the cancel queue and wait for the generation to resume
            await self.cancel_queue.put(rollout_sample)
        else:
            # put into the result_queue
            rollout_sample.param_version = self.current_param_version
            rollout_sample.rollout_status = await self.get_statistics()
            await self.result_queue.put(rollout_sample)

    async def _consumer_worker(self):
        """
        The consumer coroutine is responsible for obtaining the processing results
        from the result queue and putting them into the message queue
        """
        while True:
            rollout_sample = await self.result_queue.get()
            rollout_sample = merge_rollout_sample(self.config, self.tokenizer, rollout_sample, self.processor)

            # Put RolloutSample into the message queue
            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1

            self.result_queue.task_done()

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        # we start from step 1
        self.global_steps += 1

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine and consumer coroutine
        self.feed_task = await self.safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = await self.safe_create_task(self._processor_worker(), name="processor_task")
        self.consumer_task = await self.safe_create_task(self._consumer_worker(), name="consumer_task")

        exception_occurred = None

        try:
            await self.feed_task
            print("[FullyAsyncRollouter] Sample feed completed")
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")
            await self.consumer_task
            print("[FullyAsyncRollouter] Consumer process completed")

            await self.result_queue.join()
            await self.cancel_queue.join()

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception:{e}")
            exception_occurred = e

        finally:
            if self.feed_task:
                self.feed_task.cancel()
            if self.processor_task:
                self.processor_task.cancel()
            if self.consumer_task:
                self.consumer_task.cancel()

            await asyncio.gather(self.feed_task, self.processor_task, self.consumer_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None
            self.consumer_task = None

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False

        # Re-raise the exception after cleanup
        if exception_occurred is not None:
            raise exception_occurred

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = await self.safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = await self.safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        # wait lask validate task
        if self.validate_task:
            await self.validate_task
            self.validate_task = None

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
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
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.monitor_loop_trigger:
                if not await self._should_pause_generation():
                    async with self.lock:
                        self.paused = False
                        self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause]")
        async with self.lock:
            self.paused = True
            # Cancel all rollout tasks
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.cancel()
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
                self.active_tasks.clear()
                print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")
            await self.async_rollout_manager.reset_prefix_cache()
            self.monitor_loop_trigger = False

    async def resume(self, dependency_ref: ObjectRef = None, trigger_validate=False):
        if dependency_ref is not None:
            ray.get(dependency_ref)

        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()

            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume()

        # Async Validate
        if (
            self.val_reward_fn is not None
            and self.config.rollout.test_freq > 0
            and self.current_param_version % self.config.rollout.test_freq == 0
            and self.current_param_version > 0  # don't test here in the initial parameter sync
        ) or (self.val_reward_fn is not None and trigger_validate):
            # Create the validate asynchronous task
            self.validate_task = await self.safe_create_task(self._validate_main(), name="validate_task")

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
            "monitor/queue/result_queue_size": self.result_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/current_param_version": self.current_param_version,
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
            "monitor/validate/active_tasks_size": len(self.active_tasks_validate),
            "monitor/validate/pending_queue_size": self.pending_queue_validate.qsize(),
            "monitor/validate/cancel_queue_size": self.cancel_queue_validate.qsize(),
            "monitor/validate/result_queue_size": self.result_queue_validate.qsize(),
            "monitor/validate/total_generated_samples": self.total_generated_samples_validate,
        }

        return stats

    def _create_continuous_iterator_validate(self):
        """
        Create a continuous data iterator across epoch
        """
        iterator = iter(self.val_dataloader)
        for batch_dict in iterator:
            yield 0, batch_dict

    async def _feed_samples_validate(self):
        for epoch, batch_dict in self._create_continuous_iterator_validate():
            full_batch = prepare_single_generation_data(batch_dict, self.config.actor_rollout_ref.rollout.val_kwargs.n)
            sample_id = f"validate_sample_{epoch}_{self.global_steps_validate}"
            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                rollout_status={},
            )
            await self.pending_queue_validate.put(rollout_sample)
            self.global_steps_validate += 1

        # End signal
        await self.pending_queue_validate.put("DONE")
        print(
            f"[FullyAsyncRollouter][Validate][Feed] Sample addition is complete, {self.global_steps_validate} samples have been added"
        )

    async def _processor_worker_validate(self):
        while True:
            if self.paused:
                print(
                    "[FullyAsyncRollouter][Validate][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                while self.active_tasks_validate:
                    async with self.lock:
                        # After acquiring the lock, the number of active_tasks_validate may change, need to be verified again
                        if self.active_tasks_validate:
                            done_tasks, self.active_tasks_validate = await asyncio.wait(
                                self.active_tasks_validate, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task

                async with self.lock:
                    while self.paused:
                        await self.condition.wait()
                continue

            simple_from_cancel_queue = False
            if not self.cancel_queue_validate.empty():
                rollout_sample = await self.cancel_queue_validate.get()
                simple_from_cancel_queue = True
            else:
                rollout_sample = await self.pending_queue_validate.get()

            if rollout_sample == "DONE":
                print(
                    "[FullyAsyncRollouter][Validate][Processor] "
                    "Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks_validate:
                    async with self.lock:
                        if self.active_tasks_validate:
                            done_tasks, self.active_tasks_validate = await asyncio.wait(
                                self.active_tasks_validate, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task

                # all task success
                await self.result_queue_validate.put(None)
                break

            # Check whether the number of concurrent tasks exceeds the limit
            while len(self.active_tasks_validate) + len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks_validate:
                        done_tasks, self.active_tasks_validate = await asyncio.wait(
                            self.active_tasks_validate, return_when=asyncio.FIRST_COMPLETED
                        )
                    for task in done_tasks:
                        await task

            # Submit single sample processing
            async with self.lock:
                # After the pause is over, the lock is acquired and it is necessary
                # to determine whether it is the pause phase, otherwise continue to wait
                while self.paused:
                    await self.condition.wait()
                await self.safe_create_task(
                    self._process_single_sample_streaming_validate(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks_validate,
                )

            if simple_from_cancel_queue:
                self.cancel_queue_validate.task_done()
            else:
                self.pending_queue_validate.task_done()

    async def _process_single_sample_streaming_validate(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""

        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        agent_loop_output_list = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        rollout_sample.agent_loop_output_list = agent_loop_output_list

        is_cancel = False
        for agent_loop in agent_loop_output_list:
            if not is_cancel and agent_loop.is_cancel:
                is_cancel = True

        if is_cancel:
            # Put in the cancel queue and wait for the generation to resume
            await self.cancel_queue_validate.put(rollout_sample)
        else:
            # put into the result_queue
            await self.result_queue_validate.put(rollout_sample)

    async def _consumer_worker_validate(self):
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []

        while True:
            # Get a single sample and wait until there is a sample or None is received
            sample = await self.result_queue_validate.get()
            if sample is None:
                print(
                    f"[FullyAsyncRollouter][Consumer][Validate] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)} samples"
                )
                break
            sample = merge_rollout_sample(self.config, self.tokenizer, sample, self.processor)
            self.total_generated_samples_validate += 1
            queue_samples.append(sample)

            if len(queue_samples) % 32 == 0:
                print(f"[FullyAsyncRollouter][Consumer][Validate] Collected {len(queue_samples)} samples. ")

        consumer_end = time.time()
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncRollouter][Validate] Loop collection completed: {len(queue_samples)} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
        )
        # Assemble batch - now working directly with RolloutSample objects
        test_batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        timing_raw = {}
        val_metrics = await self._validate_metrics(test_batch)

        print(f"[FullyAsyncRollouter][Validate] {val_metrics}")
        data = ValidateMetrics(
            timing_raw=timing_raw,
            metrics=val_metrics,
            param_version=self.current_param_version,
        )
        await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))
        self.result_queue_validate.task_done()

    async def _validate_metrics(self, test_batch: DataProto):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        # we only do validation on rule-based rm
        if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            return {}

        input_ids = test_batch.batch["input_ids"]
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)

        sample_uids.extend(test_batch.non_tensor_batch["uid"])

        output_ids = test_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
        sample_gts.extend(ground_truths)

        # evaluate using reward_function
        if self.val_reward_fn is None:
            raise ValueError("val_reward_fn must be provided for validation.")
        result = self.val_reward_fn(test_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        reward_extra_infos_dict["reward"].extend(scores)
        if "reward_extra_info" in result:
            for key, lst in result["reward_extra_info"].items():
                reward_extra_infos_dict[key].extend(lst)

        # collect num_turns of each prompt
        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

        data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    async def _validate_main(self):
        if self.feed_task_validate or self.processor_task_validate or self.consumer_task_validate:
            print(f"[FullyAsyncRollouter][Validate] Validate task already running. ")

        print(
            f"[FullyAsyncRollouter][Validate] Start async validate, "
            f"maximum concurrent samples: {self.max_concurrent_samples}"
        )

        # Start sample feed coroutine, streaming process coroutine and consumer coroutine
        self.feed_task_validate = await self.safe_create_task(self._feed_samples_validate(), name="feed_task_validate")
        self.processor_task_validate = await self.safe_create_task(
            self._processor_worker_validate(), name="processor_task_validate"
        )
        self.consumer_task_validate = await self.safe_create_task(
            self._consumer_worker_validate(), name="consumer_task_validate"
        )

        exception_occurred = None
        try:
            # Wait for all tasks to complete
            await self.feed_task_validate
            await self.processor_task_validate
            await self.consumer_task_validate
            await self.result_queue_validate.join()
            await self.cancel_queue_validate.join()
        except Exception as e:
            print(f"[FullyAsyncRollouter][Validate] Streaming process exception:{e}")
            exception_occurred = e

        finally:
            print(f"[FullyAsyncRollouter][Validate] Clear Resource")
            if self.feed_task_validate:
                self.feed_task_validate.cancel()
            if self.processor_task_validate:
                self.processor_task_validate.cancel()
            if self.consumer_task_validate:
                self.consumer_task_validate.cancel()

            tasks_to_wait = []
            if self.feed_task_validate:
                tasks_to_wait.append(self.feed_task_validate)
            if self.processor_task_validate:
                tasks_to_wait.append(self.processor_task_validate)
            if self.consumer_task_validate:
                tasks_to_wait.append(self.consumer_task_validate)

            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)

            self.feed_task_validate = None
            self.processor_task_validate = None
            self.consumer_task_validate = None

        # Re-raise the exception after cleanup
        if exception_occurred is not None:
            raise exception_occurred

    async def wait_validate(self):
        if self.validate_task:
            await self.validate_task
