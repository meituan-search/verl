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
import time
from typing import Any

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer,
    TrainingStopException,
)
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking, ValidationGenerationsLogger

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10)
class TQFullyAsyncTrainer(PPOTrainer, FullyAsyncTrainer):
    """
    Fully async PPO trainer via multi-inheritance.

    - PPOTrainer: provides KVBatchMeta-native training pipeline (_compute_*, _update_*, etc.)
    - FullyAsyncTrainer: provides async infrastructure (fit loop, param sync, validate, checkpoint)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Any = None,
        device_name=None,
    ):
        # ======== 1. PPOTrainer.__init__: config, dataloader, local replay_buffer, worker groups ========
        PPOTrainer.__init__(
            self,
            config=config,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
        )
        self.tokenizer = tokenizer

        # ======== 2. FullyAsyncTrainer state fields ========
        # (mirrors FullyAsyncTrainer.__init__ lines 108-163)
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}

        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        self.rollouter = None
        self.checkpoint_manager = None
        self.hybrid_checkpoint_manager = None

        # ======== 3. TQ-specific: ReplayBuffer Ray Actor handle ========
        self.replay_buffer = None  # Set via set_replay_buffer()

        # Logger
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        print("[TQFullyAsyncTrainer] initialized (multi-inherit: PPOTrainer + FullyAsyncTrainer)")

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle."""
        self.replay_buffer = replay_buffer
        print("[TQFullyAsyncTrainer] ReplayBuffer handle set")

    # ======== Core TQ-specific: data sourcing ========
    async def _get_kvbatch_from_rb(self) -> KVBatchMeta | None:
        """
        Get a KVBatchMeta from TQ via ReplayBuffer.

        Replaces both:
        - PPOTrainer's generate_sequences() + replay_buffer.sample()
        - FullyAsyncTrainer's message_queue_client.get_sample() + assemble

        Returns KVBatchMeta compatible with PPOTrainer's entire step() pipeline.
        """
        print(f"[TQFullyAsyncTrainer] Waiting for {self.required_samples} samples from RB...", flush=True)
        consumer_start = time.time()

        sampled_keys_meta = await asyncio.wrap_future(
            self.replay_buffer.wait_and_sample.remote(
                partition_id="train",
                batch_size=self.required_samples,
            ).future()
        )

        if sampled_keys_meta is None or len(sampled_keys_meta) == 0:
            print("[TQFullyAsyncTrainer] RB returned None (termination signal)")
            return None

        keys = [k for k, _ in sampled_keys_meta]
        tags = [meta for _, meta in sampled_keys_meta]
        print(f"[TQFullyAsyncTrainer] Got {len(keys)} samples from RB")

        batch_meta = KVBatchMeta(partition_id="train", keys=keys, tags=tags)

        consumer_end = time.time()
        print(f"[TQFullyAsyncTrainer] KVBatchMeta ready: {len(keys)} keys, wait={consumer_end - consumer_start:.2f}s")

        return batch_meta

    # ======== Override: fit() — async RB consumption loop ========

    async def fit(self):
        """Main training loop: async RB consumption + PPOTrainer step() pipeline."""
        print("[TQFullyAsyncTrainer] Starting fit (KVBatchMeta pipeline)...")
        if self.replay_buffer is None:
            raise ValueError("ReplayBuffer not set. Call set_replay_buffer() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0
        self.global_steps += 1

        # Initialize TQ on this worker
        try:
            tq.init()
            print("[TQFullyAsyncTrainer] TQ initialized")
        except Exception as e:
            print(f"[TQFullyAsyncTrainer] TQ init warning: {e}")

        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[TQFullyAsyncTrainer] Training stopped by termination signal")
                break

        self.progress_bar.close()
        # Final cleanup (inherited from FullyAsyncTrainer.fit())
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """Single training step: get KVBatchMeta from RB → run PPOTrainer pipeline."""
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            # ★ CORE: Get KVBatchMeta from RB (replaces generate_sequences + replay_buffer.sample)
            batch_meta = await self._get_kvbatch_from_rb()
            if batch_meta is None:
                raise TrainingStopException("Training terminated: RB returned None")

            # Run PPOTrainer's full KVBatchMeta pipeline (steps 3-10)
            metrics = self.metrics
            timing_raw = self.timing_raw
            batch_meta.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            # Sleep rollout replicas (PPOTrainer does this after sampling)
            self.checkpoint_manager.sleep_replicas()

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                with marked_timer("reward", timing_raw, color="yellow"):
                    batch_meta = self._compute_reward_colocate(batch_meta, metrics=metrics)

            # 4. balance batch_meta across data parallel groups
            batch_meta = self._balance_batch(batch_meta, metrics=metrics)

            # 5. compute old_log_prob
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                batch_meta = self._compute_old_log_prob(batch_meta, metrics=metrics)

            # 6. [OPTIONAL] compute ref_log_prob
            if self.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    batch_meta = self._compute_ref_log_prob(batch_meta, metrics=metrics)

            # 7. [OPTIONAL] compute critic values
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    batch_meta = self._compute_values(batch_meta, metrics=metrics)

            # 8. compute advantage and return
            with marked_timer("adv", timing_raw, color="brown"):
                batch_meta = self._compute_advantage(batch_meta, metrics=metrics)

            # 9. [OPTIONAL] update critic
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    batch_meta = self._update_critic(batch_meta, metrics=metrics)

            # 10. update actor
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch_meta = self._update_actor(batch_meta, metrics=metrics)

            # Param sync every trigger_parameter_sync_step (inherited from FullyAsyncTrainer)
            self._fit_update_local_step()
            await self._fit_update_weights()

            # Cleanup consumed data from TQ + RB (TQ-specific)
            tq.kv_clear(keys=batch_meta.keys, partition_id=batch_meta.partition_id)
            ray.get(self.replay_buffer.remove.remote(batch_meta.partition_id, batch_meta.keys))

        # Validation & checkpoint (inherited from FullyAsyncTrainer)
        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch_meta)
        self._fit_postprocess_step()

    def _fit_collect_metrics(self, batch: KVBatchMeta):
        """Collect metrics using PPOTrainer's _compute_metrics (expects KVBatchMeta)."""
        self._compute_metrics(batch, self.metrics, self.timing_raw, global_steps=self.global_steps, epoch=self.epoch)
        self.logger.log(data=self.metrics, step=self.global_steps)
