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

"""
Main entry point for Elastic Scheduling in VERL

This module provides the integration between:
- FullyAsyncRollouter with elastic capabilities: Dynamic rollout resource management
- FullyAsyncTrainer with elastic capabilities: Dynamic training resource management
- ResourceCoordinator: Coordination between rollout and train
- CongestionMonitor: Monitoring production/consumption rates
"""

import asyncio
import logging
import socket
from typing import Optional

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.elastic_scheduling import (
    CongestionMonitor,
    CoordinatorLoop,
    ElasticResourceConfig,
    ElasticResourceManager,
    ResourceCoordinator,
)
from verl.experimental.elastic_scheduling.elastic_rollouter import ElasticRollouterMixin
from verl.experimental.elastic_scheduling.elastic_trainer import ElasticTrainerMixin
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class ElasticSchedulingTaskRunner:
    """
    Ray remote class for elastic scheduling PPO training.

    This extends the FullyAsyncTaskRunner with dynamic resource allocation
    between rollout and training based on congestion monitoring.

    The elastic capabilities are delegated to ElasticRollouterMixin and
    ElasticTrainerMixin which are initialized within this actor.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.coordinator_loop: Optional[CoordinatorLoop] = None

        # Initialize mixins with self reference for delegation
        self._elastic_rollouter_mixin = None
        self._elastic_trainer_mixin = None

    def run(self, config):
        """Main entry point"""
        logger.info("Starting Elastic Scheduling PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        """Initialize all components"""
        logger.info(f"TaskRunner hostname: {socket.gethostname()}")

        # Parse elastic config
        elastic_config = self._parse_elastic_config(config)

        # Initialize model and tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config
        self.components["elastic_config"] = elastic_config

        # Create resource manager
        self._init_resource_manager(elastic_config, config)

        # Create worker mapping
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # Initialize elastic mixins with config
        self._init_elastic_mixins(config)

        # Create rollouter and trainer
        self._create_rollouter_and_trainer(config)

        # Setup message queue
        self._setup_message_queue()

        # Setup coordinator
        self._setup_coordinator()

        # Load checkpoints
        self._load_checkpoints()

        # Initial parameter sync
        self._initial_sync()

    def _init_elastic_mixins(self, config) -> None:
        """Initialize elastic mixin instances with configuration"""
        # Initialize rollouter mixin
        self._elastic_rollouter_mixin = ElasticRollouterMixin()
        self._elastic_rollouter_mixin.elastic_resources = []
        self._elastic_rollouter_mixin.elastic_replicas = {}
        self._elastic_rollouter_mixin._pending_samples = None  # Will be initialized later
        self._elastic_rollouter_mixin._processing_lock = asyncio.Lock()
        self._elastic_rollouter_mixin._stats = {
            "elastic_added": 0,
            "elastic_removed": 0,
            "samples_from_elastic": 0,
        }

        # Initialize trainer mixin
        self._elastic_trainer_mixin = ElasticTrainerMixin()
        self._elastic_trainer_mixin.elastic_actors = []
        self._elastic_trainer_mixin.elastic_worker_groups = {}
        self._elastic_trainer_mixin._base_dp_size = 0
        self._elastic_trainer_mixin._stats = {
            "elastic_added": 0,
            "elastic_removed": 0,
            "batches_processed_elastic": 0,
        }
        self._elastic_trainer_mixin._sync_versions = {}
        self._elastic_trainer_mixin._current_version = 0

        logger.info("Elastic mixins initialized")

    def _parse_elastic_config(self, config) -> ElasticResourceConfig:
        """Parse elastic scheduling config from main config"""
        es_config = getattr(config, "elastic_scheduling", None)

        if es_config:
            return ElasticResourceConfig(
                rollout_gpus=es_config.get("rollout_gpus", 8),
                train_gpus=es_config.get("train_gpus", 8),
                elastic_gpus=es_config.get("elastic_gpus", 8),
                dp_size_per_resource=es_config.get("dp_size_per_resource", 8),
                rollout_queue_high_watermark=es_config.get("rollout_queue_high_watermark", 0.8),
                rollout_queue_low_watermark=es_config.get("rollout_queue_low_watermark", 0.3),
                cooldown_seconds=es_config.get("cooldown_seconds", 10.0),
                sync_trigger_interval=es_config.get("sync_trigger_interval", 4),
            )
        else:
            # Default config
            return ElasticResourceConfig(
                rollout_gpus=8,
                train_gpus=8,
                elastic_gpus=8,
                dp_size_per_resource=8,
            )

    def _init_resource_manager(
        self,
        elastic_config: ElasticResourceConfig,
        config,
    ) -> None:
        """Initialize elastic resource manager"""
        # Calculate GPU allocation
        total_gpus = elastic_config.rollout_gpus + elastic_config.train_gpus + elastic_config.elastic_gpus

        # Assuming GPUs are numbered 0 to total_gpus-1
        rollout_spec = list(range(elastic_config.rollout_gpus))
        train_spec = list(range(elastic_config.rollout_gpus, elastic_config.rollout_gpus + elastic_config.train_gpus))
        elastic_spec = list(range(elastic_config.rollout_gpus + elastic_config.train_gpus, total_gpus))

        # Create resource manager
        resource_manager = ElasticResourceManager(
            config=elastic_config,
            rollout_resource_spec=rollout_spec,
            train_resource_spec=train_spec,
            elastic_resource_spec=elastic_spec,
        )

        self.components["resource_manager"] = resource_manager
        logger.info(f"Resource manager initialized: {resource_manager.get_status_summary()}")

    def _create_rollouter_and_trainer(self, config) -> None:
        """Create rollouter and trainer with elastic support"""
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Creating Rollouter and Trainer with elastic capabilities...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Create trainer first (rollouter doesn't allow continuous allocation)
            trainer_future = executor.submit(self._create_trainer, config)
            trainer_future.result()

            rollouter_future = executor.submit(self._create_rollouter, config)
            rollouter_future.result()

        # Get shared info
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        logger.info(f"Max queue size: {max_queue_size}")

    def _create_rollouter(self, config) -> None:
        """Create rollouter with elastic support"""
        from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncRollouter

        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        # Attach elastic mixin state to rollouter actor
        self._attach_elastic_rollouter_state(rollouter)

        self.components["rollouter"] = rollouter

        # Register fixed rollout resources
        resource_manager = self.components["resource_manager"]
        for resource in resource_manager.rollout_resources:
            resource_manager.register_worker(resource.resource_id, rollouter)

    def _attach_elastic_rollouter_state(self, rollouter) -> None:
        """Attach elastic rollouter mixin state to the rollouter actor"""
        # Add elastic tracking attributes
        rollouter.elastic_resources = []
        rollouter.elastic_replicas = {}
        rollouter._elastic_stats = {
            "elastic_added": 0,
            "elastic_removed": 0,
            "samples_from_elastic": 0,
        }

    def _create_trainer(self, config) -> None:
        """Create trainer with elastic support"""
        from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTrainer

        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
        )

        ray.get(trainer.init_workers.remote())

        # Attach elastic mixin state to trainer actor
        self._attach_elastic_trainer_state(trainer)

        self.components["trainer"] = trainer

        # Register fixed train resources
        resource_manager = self.components["resource_manager"]
        for resource in resource_manager.train_resources:
            resource_manager.register_worker(resource.resource_id, trainer)

    def _attach_elastic_trainer_state(self, trainer) -> None:
        """Attach elastic trainer mixin state to the trainer actor"""
        # Add elastic tracking attributes
        trainer.elastic_actors = []
        trainer.elastic_worker_groups = {}
        trainer._elastic_stats = {
            "elastic_added": 0,
            "elastic_removed": 0,
            "batches_processed_elastic": 0,
        }
        trainer._sync_versions = {}
        trainer._current_version = 0

    def _setup_message_queue(self) -> None:
        """Setup message queue for sample passing"""
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        message_queue = MessageQueue.remote(self.components["config"], max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)

        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        # Connect to rollouter and trainer
        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))

    def _setup_coordinator(self) -> None:
        """Setup the resource coordinator"""
        elastic_config = self.components["elastic_config"]
        resource_manager = self.components["resource_manager"]

        # Create congestion monitor
        congestion_monitor = CongestionMonitor(
            window_size=20,
            high_watermark=elastic_config.rollout_queue_high_watermark,
            low_watermark=elastic_config.rollout_queue_low_watermark,
        )

        # Create coordinator
        coordinator = ResourceCoordinator(
            resource_manager=resource_manager,
            congestion_monitor=congestion_monitor,
            sync_trigger_interval=elastic_config.sync_trigger_interval,
        )

        # Setup callbacks
        coordinator.on_switch_complete = self._on_switch_complete
        coordinator.on_sync_triggered = self._on_sync_triggered

        # Create coordinator loop
        self.coordinator_loop = CoordinatorLoop(
            coordinator=coordinator,
            check_interval=elastic_config.check_interval,
        )

        self.components["coordinator"] = coordinator
        self.components["congestion_monitor"] = congestion_monitor

    async def _on_switch_complete(self, target: str, resources: list) -> None:
        """Callback when resource switch completes"""
        logger.info(f"Switched {len(resources)} resources to {target}")

        if target == "rollout":
            # Add to rollouter
            rollouter = self.components["rollouter"]
            await self._add_elastic_resources_to_rollouter(rollouter, resources)
        elif target == "train":
            # Add to trainer
            trainer = self.components["trainer"]
            await self._add_elastic_actors_to_trainer(trainer, resources)

    async def _add_elastic_resources_to_rollouter(self, rollouter, resources: list) -> None:
        """Add elastic resources to rollouter"""
        # Track resources
        rollouter.elastic_resources.extend(resources)
        rollouter._elastic_stats["elastic_added"] += len(resources)

        logger.info(f"Added {len(resources)} elastic resources to rollouter")

    async def _add_elastic_actors_to_trainer(self, trainer, resources: list) -> None:
        """Add elastic actors to trainer"""
        # Track actors
        trainer.elastic_actors.extend(resources)
        trainer._elastic_stats["elastic_added"] += len(resources)

        logger.info(f"Added {len(resources)} elastic actors to trainer")

    async def _on_sync_triggered(self, resources: list) -> None:
        """Callback when parameter sync is triggered"""
        logger.debug(f"Triggering sync to {len(resources)} resources")
        # The actual sync is handled by the checkpoint manager

    def _load_checkpoints(self) -> None:
        """Load checkpoints for resume"""
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

    def _initial_sync(self) -> None:
        """Perform initial parameter sync"""
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))
        ray.get(self.components["trainer"]._fit_update_weights.remote())

    def _run_training_loop(self):
        """Run the training loop with coordination"""
        self.running = True

        logger.info("Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        # Start coordinator loop
        if self.coordinator_loop:
            asyncio.run(self.coordinator_loop.start())

        try:
            futures = [rollouter_future, trainer_future]

            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        logger.info("Component completed successfully")
                    except Exception as e:
                        logger.error(f"Component failed: {e}")
                        for remaining in remaining_futures:
                            ray.cancel(remaining)
                        raise

                futures = remaining_futures

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Stop coordinator
            if self.coordinator_loop:
                asyncio.run(self.coordinator_loop.stop())

            # Clear queue
            asyncio.run(self.components["message_queue_client"].clear_queue())

            logger.info("Training completed or interrupted")

    def get_elastic_stats(self) -> dict:
        """Get elastic scheduling statistics"""
        rollouter = self.components.get("rollouter")
        trainer = self.components.get("trainer")

        return {
            "rollouter_elastic": {
                "elastic_resources": len(rollouter.elastic_resources) if rollouter else 0,
                "stats": rollouter._elastic_stats if rollouter else {},
            },
            "trainer_elastic": {
                "elastic_actors": len(trainer.elastic_actors) if trainer else 0,
                "stats": trainer._elastic_stats if trainer else {},
            },
            "coordinator": self.components.get("coordinator").get_current_status()
            if self.components.get("coordinator")
            else {},
        }


@hydra.main(config_path="config", config_name="elastic_ppo_trainer", version_base=None)
def main(config):
    """Main entry point"""
    from verl.trainer.main_ppo import run_ppo

    # Ensure elastic scheduling config exists
    if not hasattr(config, "elastic_scheduling"):
        # Create default config
        OmegaConf.set_struct(config, True)

        # with open_dict(config):
        #     config.elastic_scheduling = ElasticSchedulingConfig()

    # Update rollout config
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    # Run with elastic scheduling task runner
    run_ppo(config, task_runner_class=ElasticSchedulingTaskRunner)


if __name__ == "__main__":
    main()
