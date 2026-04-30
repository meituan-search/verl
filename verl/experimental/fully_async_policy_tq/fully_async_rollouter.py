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
  dataloader -> DataProto.from_single_dict -> acquire_slot -> kv_batch_put(prompt)
  -> [Worker picks up from RB] -> [Server generates] -> kv_batch_put(response) -> [Trainer consumes]
"""

import logging

import numpy as np
import ray
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.utils import need_reward_model
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
    2. Convert to DataProto (repeat is handled by AgentWorker)
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

        # Rollouter parameter configuration
        self.replay_buffer_handle = None
        self.partition_id = config.trainer.get("partition_id", "train")

        # Reward loop manager
        self.reward_loop_manager = None

        # Elastic worker group (injected before init_workers)
        self._elastic_worker_group = None

        # Statistics
        self.global_steps = 1

    def _init_async_objects(self):
        """Initialize asyncio synchronization primitives."""

    async def set_replay_buffer(self, replay_buffer_handle):
        """Set the ReplayBuffer actor handle."""
        self.replay_buffer_handle = replay_buffer_handle
        print("[TQFullyAsyncRollouter] ReplayBuffer handle set")

    def do_validate(self):
        """Run validation and return metrics."""
        pass

    def get_total_rollout_steps(self):
        """Return total rollout steps (computed from dataloader length × epochs)."""
        return self.total_rollout_steps

    def _validate_config(self):
        """Validate asynchronous training configuration."""
        if not hasattr(self.config, "async_training"):
            raise ValueError("[TQFullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate_log_probs"

    def _create_continuous_iterator(self):
        """Create a continuous data iterator across epochs."""
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _write_single_to_tq(self, batch_dict, sample_id: str):
        """Acquire slot and write a single sample's prompt data to TQ.

        Args:
            batch_dict: Raw batch dict from dataloader.
            sample_id: Unique identifier for this sample.
        """
        try:
            # Convert to DataProto
            full_batch = DataProto.from_single_dict(batch_dict)

            if not self.config.actor_rollout_ref.rollout.multi_turn.enable:
                full_batch.non_tensor_batch["agent_name"] = np.array(
                    ["single_turn_agent"] * len(full_batch), dtype=object
                )

            assert len(full_batch) == 1

            # 1. Acquire slot (blocking, backpressure control)
            acquired = await self.replay_buffer_handle.acquire_slot.remote(timeout=None)
            if not acquired:
                logging.warning("[TQFullyAsyncRollouter] Failed to acquire slot, stopping...")
                return

            # 2. Build key
            key = f"{self.partition_id}_{sample_id}"

            # 3. Build field_tensors: merge tensor + non_tensor fields, all with batch_size=1
            field_tensors = {k: t[0].unsqueeze(0) for k, t in full_batch.batch.items()}
            field_tensors.update({k: np.array([v[0]], dtype=object) for k, v in full_batch.non_tensor_batch.items()})

            # 5. Write to TQ
            tags = [{"current_status": "pending", "sample_id": sample_id}]

            print(f"[TQFullyAsyncRollouter] tags: {tags}\n\n")

            await tq.async_kv_batch_put(
                keys=[key],
                fields=TensorDict(field_tensors, batch_size=1),
                tags=tags,
                partition_id=self.partition_id,
            )

        except Exception as e:
            logging.exception(
                f"[TQFullyAsyncRollouter] Error writing sample {sample_id} to TQ: {e}full_batch: {full_batch}"
            )

    async def fit(self):
        """Start the TQ-based async rollouter: dataloader -> acquire slot -> write to TQ.

        Backpressure is handled by ReplayBuffer.acquire_slot() — when no slots are
        available, this call blocks naturally. No need for explicit concurrency control.
        """
        print("[TQFullyAsyncRollouter] Starting TQFullyAsyncRollouter...")

        if self.replay_buffer_handle is None:
            raise ValueError("ReplayBuffer not set. Call set_replay_buffer() first.")

        try:
            continuous_iterator = self._create_continuous_iterator()

            for epoch, batch_dict in continuous_iterator:
                sample_id = f"sample_{epoch}_{self.global_steps}"

                # Write to TQ (acquire_slot inside provides backpressure)
                await self._write_single_to_tq(batch_dict, sample_id)

                if self.global_steps >= self.total_rollout_steps:
                    print(
                        f"[TQFullyAsyncRollouter][FeedLoop] "
                        f"Maximum steps reached: {self.global_steps} >= {self.total_rollout_steps}"
                    )
                    break

                self.global_steps += 1

            print(f"[TQFullyAsyncRollouter][FeedLoop] Completed. {self.global_steps} samples processed")

        except Exception as e:
            print(f"[TQFullyAsyncRollouter] Error: {e}")
            raise e

        finally:
            # Signal finish to ReplayBuffer
            if self.replay_buffer_handle is not None:
                await self.replay_buffer_handle.signal_finish.remote()

        print("[TQFullyAsyncRollouter] fit completed")
