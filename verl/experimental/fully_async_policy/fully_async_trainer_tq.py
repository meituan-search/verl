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

"""TQFullyAsyncTrainer: Multi-inheritance trainer combining PPOTrainer's KVBatchMeta pipeline
with FullyAsyncTrainer's async infrastructure.

MRO: TQFullyAsyncTrainer → PPOTrainer → FullyAsyncTrainer → SeparateRayPPOTrainer → ...

Data flow:
    TQFullyAsyncRollouter --(tq.kv_batch_put)--> TransferQueue (status=finish)
        |
    TQFullyAsyncTrainer <-(RB.wait_and_sample)--+--(KVBatchMeta)--> [PPOTrainer pipeline]
                                                    |
                                              update_actor(KVBatchMeta)
"""

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
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking, ValidationGenerationsLogger

logger = logging.getLogger(__name__)

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import KVBatchMeta, tq


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
        # PPOTrainer doesn't accept device_name/ray_worker_group_cls, set them manually
        # These are required by FullyAsyncTrainer(SeparateRayPPOTrainer) init_workers pipeline:
        #   _init_resource_pools → _create_worker_classes → _init_worker_groups → _init_models
        self.device_name = device_name
        self.ray_worker_group_cls = ray_worker_group_cls or RayWorkerGroup
        self.tokenizer = tokenizer

        # Additional attributes from SeparateRayPPOTrainer/RayPPOTrainer.__init__
        # that PPOTrainer.__init__ doesn't set but _create_worker_classes / _init_models need:
        from verl.trainer.ppo.utils import need_reward_model

        self.use_rm = need_reward_model(self.config)
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

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
        self._pre_fit_reset_done = False  # Flag: main.py handles pre-fit RB.reset_staleness
        print("[TQFullyAsyncTrainer] ReplayBuffer handle set")

    # ======== Override: _fit_update_weights — skip rollouter.reset_staleness in pre-fit ========

    async def _fit_update_weights(self):
        """Override to skip rollouter.reset_staleness during pre-fit phase.

        In TQ mode, main.py calls RB.reset_staleness directly (synchronously from driver)
        to avoid a deadlock where trainer → rollouter.reset_staleness competes with
        rollouter.fit()'s event loop on the same async actor.

        After the first call (pre-fit), subsequent calls in the training loop work normally
        because rollouter's event loop is idle when reset_staleness arrives.
        """
        if self.local_trigger_step != 1:
            return

        import asyncio

        from verl.utils.debug import marked_timer

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[TQFullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}",
            flush=True,
        )

        # Skip rollouter.reset_staleness on first call (pre-fit).
        # main.py handles it via direct ray.get(rb.reset_staleness.remote()).
        if not self._pre_fit_reset_done:
            print(
                "[TQTrainer] _fit_update_weights: SKIPPING rollouter.reset_staleness (pre-fit)",
                flush=True,
            )
            self._pre_fit_reset_done = True
            return

        # Normal training-loop path: call rollouter.reset_staleness (async, safe now)
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

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

    # Override init_workers: use FullyAsyncTrainer's version (not PPOTrainer's)
    # because PPOTrainer.init_workers() assumes ActorRollout resource pool exists,
    # but in TQ fully-async mode the rollouter is a separate Ray actor.
    async def init_workers(self):
        """Initialize workers using FullyAsyncTrainer's pipeline (avoids ActorRollout pool dependency)."""
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

        # Initialize checkpoint_manager (normally done in PPOTrainer.init_workers)
        if not hasattr(self, "checkpoint_manager"):
            from verl.checkpoint_engine import CheckpointEngineManager

            self.checkpoint_manager = CheckpointEngineManager(
                config=self.config.trainer.checkpoint,
                actor_rollout_wg=getattr(self, "actor_rollout_wg", None),
                ref_wg=getattr(self, "ref_policy_wg", None),
            )

    async def fit(self):
        """Main training loop: async RB consumption + PPOTrainer step() pipeline."""
        print("[TQFullyAsyncTrainer] Starting fit (KVBatchMeta pipeline)...", flush=True)
        print(
            f"[TQTrainer] fit(): rb={self.replay_buffer is not None}, rollouter={self.rollouter is not None}",
            flush=True,
        )
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

        step_count = 0
        while True:
            try:
                print(
                    f"[TQFullyAsyncTrainer] === Starting fit_step {step_count} (global_steps={self.global_steps}) ===",
                    flush=True,
                )
                await self.fit_step()
                print(f"[TQFullyAsyncTrainer] === fit_step {step_count} DONE ===", flush=True)
                step_count += 1
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

        # Use PPOTrainer's _start_profiling (no args) instead of FullyAsyncTrainer's
        # _fit_start_profile() which passes do_profile arg incompatible with PPOTrainer
        self._start_profiling()

        with marked_timer("step", self.timing_raw):
            # ★ CORE: Get KVBatchMeta from RB (replaces generate_sequences + replay_buffer.sample)
            batch_meta = await self._get_kvbatch_from_rb()
            if batch_meta is None:
                raise TrainingStopException("Training terminated: RB returned None")

            # Run PPOTrainer's full KVBatchMeta pipeline (steps 3-10)
            metrics = self.metrics
            timing_raw = self.timing_raw
            batch_meta.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            # Abort all in-flight rollout requests BEFORE sleeping replicas.
            # This is critical for TQ fully-async mode where the rollouter is a
            # separate Ray actor that may have ongoing generate_sequences requests
            # to the SGLang server. Without aborting first, SGLang's
            # release_memory_occupation() will fail with:
            #   AssertionError: release_memory_occupation should be called only when no ongoing request
            # See: FullyAsyncTrainer._trainer_side_validate() which follows the same
            # abort → sleep → update_weights → resume pattern.
            _abort_result = self.checkpoint_manager.abort_replicas()
            if hasattr(_abort_result, "__await__"):
                await _abort_result

            # Sleep rollout replicas (PPOTrainer does this after sampling)
            # In fully async mode, rollouter is a separate actor — sleep is a no-op if
            # no rollout replicas are managed by trainer's checkpoint_manager.
            # Check if sleep_replicas is async (TQ fully-async mode) or sync (colocated mode).
            _sleep_result = self.checkpoint_manager.sleep_replicas()
            if hasattr(_sleep_result, "__await__"):
                await _sleep_result

            # 3. [OPTIONAL] compute reward score with colocated reward model
            # In TQ fully-async mode, reward is computed by Rollouter's AgentLoop
            # (via RewardLoopManager/DAPO) and stored in TQ as rm_scores.
            # Skip _compute_reward_colocate which raises NotImplementedError.
            # self.reward_loop_manager is None in TQ mode (owned by Rollouter).
            print(
                "[TQTrainer] Skipping _compute_reward_colocate (TQ mode)",
                flush=True,
            )

            # 4. balance batch_meta across data parallel groups
            batch_meta = self._balance_batch(batch_meta, metrics=metrics)

            # 5. compute old_log_prob
            # NOTE: When rollout_correction.bypass_mode=True, _compute_old_log_prob returns None
            # (missing return value in PPOTrainer's bypass path). Guard against this.
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob_result = self._compute_old_log_prob(batch_meta, metrics=metrics)
                if old_log_prob_result is not None:
                    batch_meta = old_log_prob_result
                else:
                    print(
                        "[TQTrainer] _compute_old_log_prob returned None (bypass mode)",
                        flush=True,
                    )

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

            # Wake up rollout replicas to restore GPU memory (kv_cache + weights).
            # This is the critical counterpart to sleep_replicas() called above.
            # Without wake_up, SGLang's model weights remain on CPU and subsequent
            # generate_sequences requests will fail with:
            #   ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
            _wake_result = self.checkpoint_manager.wake_up_replicas()
            if hasattr(_wake_result, "__await__"):
                await _wake_result

            # Resume generation on rollout replicas after weight sync completes.
            # This is the counterpart to abort_replicas() called above, restoring
            # the rollouter's ability to send generate_sequences requests to SGLang.
            _resume_result = self.checkpoint_manager.resume_generation_replicas()
            if hasattr(_resume_result, "__await__"):
                await _resume_result

            # Cleanup consumed data from TQ + RB (TQ-specific)
            tq.kv_clear(keys=batch_meta.keys, partition_id=batch_meta.partition_id)
            ray.get(self.replay_buffer.remove.remote(batch_meta.partition_id, batch_meta.keys))

        # Validation & checkpoint (inherited from FullyAsyncTrainer)
        await self._fit_validate()
        self._fit_save_checkpoint()
        # Use PPOTrainer's _stop_profiling (no args) instead of FullyAsyncTrainer's
        self._stop_profiling()
        self._fit_collect_metrics(batch_meta)
        self._fit_postprocess_step()

    def _fit_collect_metrics(self, batch: KVBatchMeta):
        """Collect metrics using PPOTrainer's _compute_metrics (expects KVBatchMeta)."""
        self._compute_metrics(batch, self.metrics, self.timing_raw, global_steps=self.global_steps, epoch=self.epoch)
        self.logger.log(data=self.metrics, step=self.global_steps)
