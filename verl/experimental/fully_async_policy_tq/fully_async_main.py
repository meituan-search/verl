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

"""Main entry point for TQ-based fully async PPO training.

Architecture Overview:
  1. TQFullyAsyncRollouter: Reads dataloader -> writes prompts to TQ (with slot backpressure)
  2. TQAgentLoopWorker: Pulls tasks from RB -> dispatches to LLM Servers via TQ
  3. TQLLMServer: Reads prompt from TQ -> generates -> writes response to TQ
  4. TQFullyAsyncTrainer: Consumes finished samples from RB/TQ -> trains

Key differences from fully_async_main.py (MessageQueue):
  - Uses TransferQueue instead of MessageQueue for data transport
  - Uses ReplayBuffer (TQ version) instead of MessageQueue for signaling
  - Workers actively pull tasks instead of being dispatched by AgentLoopManager
  - Zero-copy tensor transport via TQ kv_batch_put/kv_batch_get
"""

import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy_tq.fully_async_rollouter import TQFullyAsyncRollouter
from verl.experimental.fully_async_policy_tq.fully_async_trainer import TQFullyAsyncTrainer
from verl.experimental.fully_async_policy_tq.replay_buffer import ReplayBuffer
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class TQFullyAsyncTaskRunner:
    """Ray remote class for executing TQ-based distributed PPO training."""

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        """Main entry point called by Ray."""
        print("[TQ_ASYNC MAIN] Starting TQ-based fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        """Initialize all components for TQ-based training."""
        print(f"[TQ_ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ==================== 1. Initialize TransferQueue ====================
        print("[TQ_ASYNC MAIN] Initializing TransferQueue...")
        try:
            import transfer_queue as tq

            tq_config = getattr(config, "transfer_queue", None)
            if tq_config:
                tq.init(OmegaConf.to_container(tq_config, resolve=True))
            else:
                tq.init()
            print("[TQ_ASYNC MAIN] TransferQueue initialized successfully")
        except ImportError as err:
            raise RuntimeError("TransferQueue not installed. Please run: pip install TransferQueue==0.1.6") from err

        # ==================== 2. Initialize tokenizer and processor ====================
        print("[TQ_ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        # ==================== 3. Create worker mapping and resource pools ====================
        print("[TQ_ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # ==================== 4. Create ReplayBuffer ====================
        max_pending_slots = config.async_training.get("max_pending_slots", 256)
        poll_interval = config.async_training.get("poll_interval", 1.0)
        print(f"[TQ_ASYNC MAIN] Creating ReplayBuffer (max_slots={max_pending_slots}, poll={poll_interval}s)"
        replay_buffer = ReplayBuffer.remote(
            max_pending_slots=max_pending_slots,
            poll_interval=poll_interval,
        )
        self.components["replay_buffer"] = replay_buffer

        # ==================== 5. Create Trainer first (needed for elastic worker group injection) ====================
        print("[TQ_ASYNC MAIN] Creating TQFullyAsyncTrainer first...")
        self._create_trainer(config)

        # ==================== 6. Setup elastic worker group ====================
        print("[TQ_ASYNC MAIN] Setting up elastic worker group...")
        self._setup_elastic_worker_group(config)

        # ==================== 7. Create Rollouter ====================
        print("[TQ_ASYNC MAIN] Creating TQFullyAsyncRollouter...")
        self._create_rollouter(config)

        # ==================== 8. Wire up references ====================
        print("[TQ_ASYNC MAIN] Wiring up component references...")

        # Set ReplayBuffer on both rollouter and trainer
        ray.get(self.components["rollouter"].set_replay_buffer.remote(replay_buffer))
        ray.get(self.components["trainer"].set_replay_buffer.remote(replay_buffer))

        # Set rollouter reference on trainer (for param sync, validation, checkpointing)
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # Sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"[TQ_ASYNC MAIN] total_train_steps: {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # Initialize max_required_samples on rollouter
        ray.get(self.components["rollouter"].set_max_required_samples.remote())

        # ==================== 9. Create and start TQAgentLoopWorkers ====================
        print("[TQ_ASYNC MAIN] Creating TQAgentLoopWorkers...")
        self._create_and_start_workers(config)

        # ==================== 10. Load checkpoints ====================
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        # ==================== 11. Initial parameter sync ====================
        print("[TQ_ASYNC MAIN] Initial parameter sync before fit...")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        # ==================== 12. Optional pre-training validation ====================
        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[TQ_ASYNC MAIN] All components initialized successfully")

    def _create_trainer(self, config) -> None:
        """Create the TQFullyAsyncTrainer."""
        print("[TQ_ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = TQFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[TQ_ASYNC MAIN] TQFullyAsyncTrainer created and initialized")

    def _create_rollouter(self, config) -> None:
        """Create the TQFullyAsyncRollouter."""
        print("[TQ_ASYNC MAIN] Starting create rollouter...")
        rollouter = TQFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        # Inject elastic worker group before init_workers
        if "elastic_worker_group" in self.components:
            ray.get(rollouter.set_elastic_worker_group.remote(self.components["elastic_worker_group"]))
            print("[TQ_ASYNC MAIN] Elastic worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())

        self.components["rollouter"] = rollouter
        print("[TQ_ASYNC MAIN] TQFullyAsyncRollouter created and initialized")

    def _setup_elastic_worker_group(self, config) -> None:
        """Extract trainer's actor_wg as elastic worker group for rollouter validation."""
        trainer = self.components["trainer"]
        if config.async_training.use_trainer_do_validate:
            trainer_wg = ray.get(trainer.get_actor_wg.remote())
            self.components["elastic_worker_group"] = trainer_wg
            print(
                f"[TQ_ASYNC MAIN] Elastic worker group extracted (world_size={getattr(trainer_wg, 'world_size', '?')})"
            )
        else:
            print("[TQ_ASYNC MAIN] use_trainer_do_validate=False, skipping elastic setup")

    def _create_and_start_workers(self, config) -> None:
        """Create TQAgentLoopWorkers and start their event loops.

        In the TQ architecture, workers are independent actors that:
        1. Actively pull tasks from ReplayBuffer
        2. Create AgentLoop instances for each task
        3. Dispatch generation requests to LLM Servers via TQ
        """
        from verl.experimental.fully_async_policy_tq.agent_loop import TQAgentLoopWorker

        # We need server handles and load balancer from an AgentLoopManager.
        # For now, we create a minimal ALM just to get its servers/LB infrastructure.
        # The actual rollout replicas are managed by the rollouter's internal ALM.

        num_workers = config.actor_rollout_ref.rollout.agent.get("num_workers", 2)
        print(f"[TQ_ASYNC MAIN] Creating {num_workers} TQAgentLoopWorkers...")

        # Get server addresses and handles from the rollouter's internal rollout manager.
        # Note: In a full implementation, we would extract these from the rollouter's
        # FullyAsyncAgentLoopManager after it's been initialized.
        # For now, workers will be started but need server handles injected later.

        self.components["tq_workers"] = []
        self.components["worker_handles"] = []

        for i in range(num_workers):
            worker = TQAgentLoopWorker.remote(
                config=config,
                replay_buffer_handle=self.components["replay_buffer"],
                servers=[],  # Will be populated via add_server calls
                load_balancer_handle=None,  # Will be set after LB creation
                tokenizer=self.components["tokenizer"],
                processor=self.components["processor"],
            )
            self.components["tq_workers"].append(worker)
            self.components["worker_handles"].append(worker)

            # Start the event loop
            ray.get(worker.start.remote())
            print(f"[TQ_ASYNC MAIN] TQAgentLoopWorker-{i} started")

        print(f"[TQ_ASYNC MAIN] All {num_workers} TQAgentLoopWorkers created and started")
        print("[TQ_ASYNC MAIN] NOTE: Server handles need to be registered with workers after rollout init")

    def _run_training_loop(self):
        """Run the main training loop: start rollouter and trainer concurrently."""
        self.running = True

        print("[TQ_ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[TQ_ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[TQ_ASYNC MAIN] Component failed: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[TQ_ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            # Cleanup: stop all workers
            print("[TQ_ASYNC MAIN] Cleaning up...")
            self._cleanup()

    def _cleanup(self):
        """Clean up resources on shutdown."""
        # Signal finish to all TQAgentLoopWorkers
        if "tq_workers" in self.components:
            for worker in self.components["tq_workers"]:
                try:
                    ray.get(worker.signal_finish.remote())
                except Exception as e:
                    print(f"[TQ_ASYNC MAIN] Error signaling worker finish: {e}")

        # Signal finish to ReplayBuffer
        if "replay_buffer" in self.components:
            try:
                ray.get(self.components["replay_buffer"].signal_finish.remote())
            except Exception as e:
                print(f"[TQ_ASYNC MAIN] Error signaling RB finish: {e}")

        print("[TQ_ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    """Main entry point for TQ-based fully async PPO training."""
    from verl.trainer.main_ppo import run_ppo

    # Ensure async_training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)

    # Enable TransferQueue
    config.transfer_queue.enable = True

    run_ppo(config, task_runner_class=TQFullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
