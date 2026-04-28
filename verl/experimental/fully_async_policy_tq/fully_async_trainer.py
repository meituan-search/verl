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

"""TQFullyAsyncTrainer: Consumes finished samples from TQ for PPO training.

Key differences from FullyAsyncTrainer (MessageQueue-based):
- Gets samples via ReplayBuffer.wait_and_sample() instead of MessageQueue.get_sample()
- Reads full data from TQ via kv_batch_get() instead of unpickling RolloutSample
- No pickle serialization/deserialization overhead

Data Flow:
  Trainer -> RB.wait_and_sample(finish keys) -> TQ.kv_batch_get(full data)
  -> PPO training step -> TQ.kv_clear(consumed data) -> RB.remove(keys)
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop."""

    pass


@ray.remote(num_cpus=10)
class TQFullyAsyncTrainer(SeparateRayPPOTrainer):
    """Fully asynchronous PPO trainer that consumes samples from TQ via ReplayBuffer."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # ==================== Base config ====================
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.config)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
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
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async TQ config ====================
        self.replay_buffer_handle = None  # Set via set_replay_buffer()

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
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

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation
        self.hybrid_checkpoint_manager = None

    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized."""
        replicas = ray.get(self.rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[TQFullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager for trainer-side validation."""
        if not self.config.async_training.use_trainer_do_validate:
            return

        print("[TQFullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=[],
        )

        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        elastic_replicas_dict = ray.get(self.rollouter.get_all_elastic_replicas.remote())
        print(f"[TQFullyAsyncTrainer] Got {len(elastic_replicas_dict)} elastic replicas")

        if not elastic_replicas_dict:
            return

        for resource_id, replica in elastic_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)

        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[TQFullyAsyncTrainer] Initial sleep complete")

    async def set_replay_buffer(self, replay_buffer_handle):
        """Set ReplayBuffer actor handle."""
        self.replay_buffer_handle = replay_buffer_handle
        print("[TQFullyAsyncTrainer] ReplayBuffer handle set")

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize checkpoint managers."""
        self.rollouter = rollouter
        self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """Get training samples from TQ via ReplayBuffer.

        Returns:
            tuple: (epoch, batch) or (None, None) if no more samples
        """
        partition_id = "train"
        print(
            f"[TQFullyAsyncTrainer] Requesting {self.required_samples} samples from TQ",
            flush=True,
        )

        consumer_start = time.time()

        # 1. Wait for enough finished samples from ReplayBuffer
        samples = await self.replay_buffer_handle.wait_and_sample.remote(
            partition_id=partition_id,
            batch_size=self.required_samples,
        )

        if samples is None:
            print("[TQFullyAsyncTrainer] No more samples available")
            return None, None

        keys = [k for k, _ in samples]
        metas = [v for _, v in samples]

        if len(keys) < self.required_samples:
            print(f"[TQFullyAsyncTrainer] Only got {len(keys)}/{self.required_samples} samples")
            if len(keys) == 0:
                return None, None

        consumer_end = time.time()
        total_wait_time = consumer_end - consumer_start

        print(f"[TQFullyAsyncTrainer] Got {len(keys)} samples, wait time: {total_wait_time:.2f}s")

        # 2. Get full data from TQ
        fields = [
            "input_ids",
            "response_ids",
            "response_mask",
            "rollout_log_probs",
            "rm_scores",
            "attention_mask",
            "position_ids",
            "prompts",
            "responses",
        ]

        try:
            tq_data = await tq.async_kv_batch_get(
                keys=keys,
                select_fields=fields,
                partition_id=partition_id,
            )
        except Exception as e:
            logger.exception(f"[TQFullyAsyncTrainer] Error reading from TQ: {e}")
            return None, None

        # 3. Build DataProto batch from TQ data
        batch = self._build_data_proto_from_tq(tq_data, keys, metas)
        if batch is None:
            return None, None

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time

        # 4. Calculate staleness statistics
        staleness_stats = {
            "staleness/mean": float(
                np.mean(
                    [
                        self.current_param_version - m.get("start_model_version", self.current_param_version)
                        for m in metas
                    ]
                )
            ),
            "staleness/max": max(
                [self.current_param_version - m.get("start_model_version", self.current_param_version) for m in metas]
            ),
        }
        batch.meta_info.update(staleness_stats)

        return 0, batch

    def _build_data_proto_from_tq(self, tq_data: dict, keys: list, metas: list) -> DataProto | None:
        """Build a DataProto batch from TQ data.

        Args:
            tq_data: Dictionary of field_name -> list of tensors from TQ
            keys: List of sample keys
            metas: List of metadata dicts

        Returns:
            Assembled DataProto batch
        """
        if not tq_data:
            logger.warning("[TQFullyAsyncTrainer] Empty TQ data")
            return None

        batch_tensors = {}
        non_tensor_batch = {}

        # Process each field
        for field_name, values in tq_data.items():
            if not values:
                continue

            if isinstance(values[0], torch.Tensor):
                # Tensor field - stack into batch tensor
                try:
                    batch_tensors[field_name] = torch.stack(values, dim=0)
                except Exception as e:
                    logger.warning(f"[TQFullyAsyncTrainer] Could not stack {field_name}: {e}")
                    # Try padding if shapes differ
                    padded = self._pad_tensors(values)
                    if padded is not None:
                        batch_tensors[field_name] = padded
            elif isinstance(values[0], str | int | float | np.ndarray):
                # Non-tensor field
                non_tensor_batch[field_name] = np.array(values, dtype=object)

        # Build non-tensor metadata from tags
        meta_keys_to_extract = [
            "uid",
            "session_id",
            "trajectory_id",
            "start_model_version",
            "end_model_version",
            "prompt_len",
            "response_len",
            "seq_len",
            "min_global_steps",
            "max_global_steps",
            "data_source",
            "agent_name",
        ]
        for mk in meta_keys_to_extract:
            if mk not in non_tensor_batch:
                vals = [m.get(mk, "") for m in metas]
                non_tensor_batch[mk] = np.array(vals, dtype=object)

        # Ensure required fields exist
        if "input_ids" not in batch_tensors:
            logger.error("[TQFullyAsyncTrainer] Missing input_ids in TQ data")
            return None

        # Compute response_mask if missing
        if "response_mask" not in batch_tensors:
            from verl.trainer.ppo.ray_trainer import compute_response_mask

            # Create temporary batch for mask computation
            temp_batch = DataProto(batch=batch_tensors, non_tensor_batch=non_tensor_batch)
            batch_tensors["response_mask"] = compute_response_mask(temp_batch)

        # Add trajectory param versions for staleness tracking
        non_tensor_batch["min_global_steps"] = np.array(
            [m.get("start_model_version", 0) for m in metas], dtype=np.int64
        )
        non_tensor_batch["max_global_steps"] = np.array([m.get("end_model_version", 0) for m in metas], dtype=np.int64)

        batch = DataProto(batch=batch_tensors, non_tensor_batch=non_tensor_batch)

        # Compute global token count
        if "attention_mask" in batch.batch:
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return batch

    def _pad_tensors(self, tensors: list[torch.Tensor]) -> torch.Tensor | None:
        """Pad tensors of different lengths to same size for stacking."""
        if not tensors:
            return None

        max_len = max(t.shape[0] for t in tensors)
        dtype = tensors[0].dtype

        padded = []
        for t in tensors:
            if t.shape[0] < max_len:
                pad = torch.zeros(max_len - t.shape[0], dtype=dtype)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)

        return torch.stack(padded, dim=0)

    def _create_actor_rollout_classes(self):
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    async def init_workers(self):
        """Initialize distributed training workers."""
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """Main training loop."""
        print("[TQFullyAsyncTrainer] Starting TQFullyAsyncTrainer...")
        if self.replay_buffer_handle is None:
            raise ValueError("ReplayBuffer not set. Call set_replay_buffer() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0
        self.global_steps += 1
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[TQFullyAsyncTrainer] Training stopped")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """Single training step."""
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: no more samples")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """Compute old log prob with version-aware parameter management."""
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[TQFullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[TQFullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f}s "
            f"current_param_version: {self.current_param_version}"
        )

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(data=timing_raw, step=self.current_param_version)

        # Log aggregated metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        if not need_validate and not val_before_train:
            return

        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = ray.get(self.rollouter.do_validate.remote())
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using elastic rollout replicas."""
        print("[TQFullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()

        # Phase 1: Switch GPUs to ROLLOUT mode
        phase_1_start = time.time()
        print("[TQFullyAsyncTrainer] Phase 1: Switching GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        elastic_replicas_dict = ray.get(self.rollouter.get_all_elastic_replicas.remote())
        elastic_resource_ids = list(elastic_replicas_dict.keys())
        for resource_id in elastic_resource_ids:
            ray.get(self.rollouter.add_elastic_replica.remote(resource_id))
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[TQFullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # Phase 2: Run validation
        print("[TQFullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = ray.get(self.rollouter.do_validate.remote())
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # Phase 3: Switch back to TRAIN mode
        print("[TQFullyAsyncTrainer] Phase 3: Switching back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        for resource_id in elastic_resource_ids:
            ray.get(self.rollouter.remove_elastic_replica.remote(resource_id))
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[TQFullyAsyncTrainer] _trainer_side_validate === END === ({total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1
        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )
        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )
        print(f"[TQFullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("[TQFullyAsyncTrainer] Warning: remove_previous_ckpt_in_save deprecated")
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs not implemented")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str"
                assert "global_step_" in self.config.trainer.resume_from_path, "must specify global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    global_step_folder = os.path.join(os.getcwd(), global_step_folder)
            else:
                raise ValueError(f"Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[TQFullyAsyncTrainer] Loading from: {global_step_folder}")
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[TQFullyAsyncTrainer] global_steps={self.global_steps}, "
            f"current_param_version={self.current_param_version}"
        )

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """Collect metrics from training batch."""
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info.get("trajectory_param_versions", [])
            if trajectory_param_versions is not None:
                stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
                self.stale_trajectory_processed += stale_traj_count
                metrics.update(
                    {
                        "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                        "fully_async/count/current_param_version": self.current_param_version,
                    }
                )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value
