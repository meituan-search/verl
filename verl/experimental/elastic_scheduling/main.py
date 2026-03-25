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
Elastic Scheduling Main Entry Point for VERL

This module wires together all elastic scheduling components:
- ElasticRollouter: Extends FullyAsyncRollouter with dynamic rollout replica management
- ElasticTrainer: Extends FullyAsyncTrainer with dynamic training DP management
- ElasticCoordinator: Monitors production/consumption rates and triggers role switches
- ElasticParameterSyncManager: Handles parameter sync to both fixed and elastic replicas
- MessageQueue: The shared buffer between rollouter and trainer

Architecture:
    ElasticSchedulingTaskRunner (Ray actor)
        ├── ElasticRollouter (Ray actor)
        │     ├── ElasticAgentLoopManager
        │     │     ├── ElasticGlobalRequestLoadBalancer
        │     │     └── AgentLoopWorkers (dynamic)
        │     └── Elastic Replicas (managed by coordinator)
        ├── ElasticTrainer (Ray actor)
        │     └── Elastic Actor WGs (managed by coordinator)
        ├── MessageQueue (Ray actor, shared buffer)
        ├── ElasticCoordinator (Ray actor)
        │     ├── Polls queue size from MessageQueue
        │     ├── Polls rates from Rollouter/Trainer
        │     └── Switches elastic resources between modes
        └── ElasticParameterSyncManager (local in trainer process)
              ├── Fixed replicas via CheckpointEngineManager
              └── Elastic replicas via additional sync calls

Usage:
    python -m verl.experimental.elastic_scheduling.main \
        --config-path config \
        --config-name elastic_ppo_trainer

    Or with OmegaConf override:
        trainer.nnodes=2 trainer.n_gpus_per_node=8 \
        rollout.nnodes=1 rollout.n_gpus_per_node=8 \
        elastic_scheduling.elastic_nnodes=1 elastic_scheduling.elastic_n_gpus_per_node=8
"""

import asyncio
import logging
import socket
from typing import Optional

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)


# ============================================================================
# Elastic Scheduling Task Runner
# ============================================================================


@ray.remote(num_cpus=4)
class ElasticSchedulingTaskRunner:
    """
    Ray remote class for elastic scheduling PPO training.

    This creates and manages:
    1. ElasticRollouter: async sample generation with dynamic replica pool
    2. ElasticTrainer: async training with dynamic DP pool
    3. MessageQueue: FIFO buffer between rollouter and trainer
    4. ElasticCoordinator: monitoring + role switching loop
    5. ElasticParameterSyncManager: handles param sync to all replicas

    The ElasticCoordinator acts as the brain of the system, continuously:
    - Monitoring queue utilization (production vs consumption rates)
    - Deciding when to switch elastic resources between rollout and train modes
    - Ensuring role switches happen BEFORE parameter sync cycles
    - Coordinating DP group rebuilds on train workers

    Elastic Resource Configuration:
        Fixed rollout resources: Always in rollout mode (dedicated inference)
        Fixed train resources: Always in train mode (dedicated training)
        Elastic resources: Can switch between rollout and train modes
                          (initialized as either rollout or train based on config)
    """

    def __init__(self):
        self.running = False
        self.components: dict = {}
        self._coordinator_task: Optional[asyncio.Task] = None

    def run(self, config: DictConfig):
        """Main entry point. Initializes all components and starts training."""
        logger.info(f"[ElasticSchedulingTaskRunner] Starting on {socket.gethostname()}")

        # Initialize all components
        self._initialize_components(config)

        # Run training loop (blocking)
        self._run_training_loop()

    # =========================================================================
    # Component Initialization
    # =========================================================================

    def _initialize_components(self, config: DictConfig):
        """Initialize all components in the correct order."""
        logger.info("[ElasticSchedulingTaskRunner] Initializing components...")

        # 1. Load tokenizer and processor
        self._init_tokenizer(config)

        # 2. Create resource pool managers and role-worker mappings
        self._init_resource_mappings(config)

        # 3. Create ElasticTrainer and ElasticRollouter
        self._init_trainer_and_rollouter(config)

        # 4. Setup MessageQueue
        self._init_message_queue(config)

        # 5. Setup ElasticCoordinator
        self._init_coordinator(config)

        # 6. Connect ElasticParameterSyncManager (wraps trainer's checkpoint manager)
        self._init_elastic_param_sync(config)

        # 7. Load checkpoints
        self._load_checkpoints()

        # 8. Initial parameter sync
        self._initial_param_sync()

        logger.info("[ElasticSchedulingTaskRunner] All components initialized")

    def _init_tokenizer(self, config: DictConfig):
        """Load model tokenizer and processor."""
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=False)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config
        logger.info("[ElasticSchedulingTaskRunner] Tokenizer loaded")

    def _init_resource_mappings(self, config: DictConfig):
        """Create role-worker mappings for rollouter and trainer."""

        # Try to use separation utils if available
        try:
            from verl.experimental.separation.utils import (
                create_resource_pool_manager,
                create_role_worker_mapping,
            )

            role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
            self.components["role_worker_mapping"] = role_worker_mapping
            self.components["ray_worker_group_cls"] = ray_worker_group_cls
            self.components["create_resource_pool_manager"] = create_resource_pool_manager

        except ImportError:
            logger.warning("[ElasticSchedulingTaskRunner] separation.utils not available, using defaults")
            from verl.single_controller.ray import RayWorkerGroup

            self.components["ray_worker_group_cls"] = RayWorkerGroup
            self.components["role_worker_mapping"] = {}
            self.components["create_resource_pool_manager"] = None

    def _init_trainer_and_rollouter(self, config: DictConfig):
        """Create ElasticTrainer and ElasticRollouter sequentially."""

        logger.info("[ElasticSchedulingTaskRunner] Creating ElasticTrainer and ElasticRollouter...")

        # Create trainer first (it needs more resources and must reserve GPUs)
        self._create_elastic_trainer(config)
        # Then create rollouter (uses remaining resources)
        self._create_elastic_rollouter(config)

        logger.info("[ElasticSchedulingTaskRunner] ElasticTrainer and ElasticRollouter created")

    def _create_elastic_trainer(self, config: DictConfig):
        """Create ElasticTrainer Ray actor."""
        from verl.experimental.elastic_scheduling.elastic_trainer import ElasticTrainer
        from verl.trainer.ppo.utils import Role

        create_rp = self.components.get("create_resource_pool_manager")
        role_worker_mapping = self.components.get("role_worker_mapping", {})
        ray_worker_group_cls = self.components.get("ray_worker_group_cls")

        # Trainer roles: Actor (training), optionally Critic and Ref
        trainer_roles = {role: wcls for role, wcls in role_worker_mapping.items() if role != Role.Rollout}

        trainer_resource_pool_manager = create_rp(config, roles=list(trainer_roles.keys())) if create_rp else None

        trainer = ElasticTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_roles,
            resource_pool_manager=trainer_resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=self.components["processor"],
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        logger.info("[ElasticSchedulingTaskRunner] ElasticTrainer created and workers initialized")

    def _create_elastic_rollouter(self, config: DictConfig):
        """Create ElasticRollouter Ray actor."""
        from verl.experimental.elastic_scheduling.elastic_rollouter import ElasticRollouter
        from verl.trainer.ppo.utils import Role

        create_rp = self.components.get("create_resource_pool_manager")
        role_worker_mapping = self.components.get("role_worker_mapping", {})
        ray_worker_group_cls = self.components.get("ray_worker_group_cls")

        rollouter_resource_pool_manager = create_rp(config, roles=[Role.Rollout]) if create_rp else None

        rollouter = ElasticRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=rollouter_resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=self.components["processor"],
        )

        ray.get(rollouter.init_workers.remote())
        # Note: set_max_required_samples requires message_queue_client to be set first
        # This will be called from _init_message_queue()
        self.components["rollouter"] = rollouter
        logger.info("[ElasticSchedulingTaskRunner] ElasticRollouter created and workers initialized")

    def _init_message_queue(self, config: DictConfig):
        """Setup MessageQueue and connect to rollouter and trainer."""
        from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient

        # First set up max_required_samples to determine queue size
        ray.get(self.components["rollouter"].set_max_required_samples.remote())
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())

        # Synchronize total train steps from rollouter to trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))
        logger.info(
            f"[ElasticSchedulingTaskRunner] total_train_steps={total_train_steps}, max_queue_size={max_queue_size}"
        )

        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)

        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        # Connect queue to rollouter and trainer
        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))

        logger.info(f"[ElasticSchedulingTaskRunner] MessageQueue initialized (capacity={max_queue_size})")

    def _init_coordinator(self, config: DictConfig):
        """
        Setup ElasticCoordinator.

        Parses elastic resource info from config and creates coordinator.
        The coordinator is responsible for monitoring and switching elastic resources.
        """
        from verl.experimental.elastic_scheduling.coordinator import ElasticCoordinator

        elastic_config = getattr(config, "elastic_scheduling", {})

        # Parse elastic resource info
        # In a real deployment, worker handles would come from the resource pool manager
        # For now, we initialize with empty list and resources can be registered later
        elastic_resource_infos = self._build_elastic_resource_infos(config)

        coordinator_config = {
            "high_watermark": float(getattr(elastic_config, "rollout_queue_high_watermark", 0.8)),
            "low_watermark": float(getattr(elastic_config, "rollout_queue_low_watermark", 0.3)),
            "cooldown_seconds": float(getattr(elastic_config, "cooldown_seconds", 30.0)),
            "check_interval": float(getattr(elastic_config, "check_interval", 2.0)),
            "ema_alpha": float(getattr(elastic_config, "ema_alpha", 0.3)),
            "min_rollout_resources": int(getattr(elastic_config, "min_rollout_resources", 0)),
            "min_train_resources": int(getattr(elastic_config, "min_train_resources", 0)),
            "confidence_threshold": float(getattr(elastic_config, "confidence_threshold", 0.6)),
            "max_concurrent_switches": int(getattr(elastic_config, "max_concurrent_switches", 1)),
        }

        coordinator = ElasticCoordinator.remote(
            elastic_rollouter=self.components["rollouter"],
            elastic_trainer=self.components["trainer"],
            message_queue=self.components["message_queue"],
            elastic_resource_infos=elastic_resource_infos,
            config=coordinator_config,
        )

        self.components["coordinator"] = coordinator

        # Wire coordinator to trainer for pre-sync hook
        ray.get(self.components["trainer"].set_elastic_coordinator.remote(coordinator))

        logger.info(
            f"[ElasticSchedulingTaskRunner] ElasticCoordinator initialized with "
            f"{len(elastic_resource_infos)} elastic resources"
        )

    def _build_elastic_resource_infos(self, config: DictConfig) -> list:
        """
        Build elastic resource info list from config.

        In production, this would query the resource pool manager for
        the elastic worker handles. For now returns an empty list.
        """
        elastic_config = getattr(config, "elastic_scheduling", {})
        n_elastic = int(getattr(elastic_config, "n_elastic_resources", 0))

        # TODO: Get actual worker handles from resource pool manager
        # For now, return placeholder info
        elastic_resource_infos = []
        for i in range(n_elastic):
            elastic_resource_infos.append(
                {
                    "resource_id": f"elastic_{i}",
                    "initial_mode": getattr(elastic_config, "elastic_initial_mode", "rollout"),
                    "worker_handles": [],  # Will be populated after workers are initialized
                }
            )

        return elastic_resource_infos

    def _init_elastic_param_sync(self, config: DictConfig):
        """
        Setup ElasticParameterSyncManager.

        This wraps the CheckpointEngineManager to also handle elastic replicas.
        The manager is created inside the trainer process (not as a separate Ray actor).
        """

        # Connect rollouter to trainer (triggers checkpoint manager initialization)
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # Note: ElasticParameterSyncManager is used inside the trainer process
        # The trainer's _fit_update_weights() will use it automatically
        # when self.checkpoint_manager is set to an ElasticParameterSyncManager

        logger.info("[ElasticSchedulingTaskRunner] ElasticParameterSyncManager initialized")

    def _load_checkpoints(self):
        """Load checkpoints for resume training."""
        logger.info("[ElasticSchedulingTaskRunner] Loading checkpoints...")
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

    def _initial_param_sync(self):
        """Perform initial parameter sync from trainer to rollouter."""
        logger.info("[ElasticSchedulingTaskRunner] Performing initial parameter sync...")
        # _fit_update_weights is an async Ray method - call it via ray.get()
        # This syncs trainer weights to rollout replicas before training starts
        try:
            ray.get(self.components["trainer"]._fit_update_weights.remote())
            logger.info("[ElasticSchedulingTaskRunner] Initial parameter sync complete")
        except Exception as e:
            logger.warning(
                f"[ElasticSchedulingTaskRunner] Initial param sync failed (may be OK if checkpoint loaded): {e}"
            )

    # =========================================================================
    # Training Loop
    # =========================================================================

    def _run_training_loop(self):
        """
        Run the main training loop.

        Starts rollouter, trainer, and coordinator concurrently.
        Waits for completion or handles failures.
        """
        self.running = True
        logger.info("[ElasticSchedulingTaskRunner] Starting training loop...")

        # Start coordinator (async background task, fire-and-forget)
        self.components["coordinator"].start.remote()

        # Start rollouter and trainer
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        try:
            # Wait for rollouter and trainer to complete
            futures = [rollouter_future, trainer_future]

            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        logger.info("[ElasticSchedulingTaskRunner] Component completed successfully")
                    except Exception as e:
                        logger.error(f"[ElasticSchedulingTaskRunner] Component failed: {e}")
                        # Cancel remaining futures
                        for remaining in remaining_futures:
                            ray.cancel(remaining)
                        raise

                futures = remaining_futures

        except Exception as e:
            logger.error(f"[ElasticSchedulingTaskRunner] Training failed: {e}")
            raise

        finally:
            # Stop coordinator
            logger.info("[ElasticSchedulingTaskRunner] Stopping coordinator...")
            ray.get(self.components["coordinator"].stop.remote())

            # Clear message queue
            message_queue_client = self.components.get("message_queue_client")
            if message_queue_client:
                asyncio.run(message_queue_client.clear_queue())

            self.running = False
            logger.info("[ElasticSchedulingTaskRunner] Training completed")

    # =========================================================================
    # Monitoring / Statistics
    # =========================================================================

    def get_status(self) -> dict:
        """Get current status of all components."""
        status = {
            "running": self.running,
        }

        try:
            coordinator_status = ray.get(self.components["coordinator"].get_status.remote())
            status["coordinator"] = coordinator_status
        except Exception:
            pass

        try:
            elastic_stats = ray.get(self.components["trainer"].get_elastic_statistics.remote())
            status["trainer"] = elastic_stats
        except Exception:
            pass

        try:
            elastic_stats = ray.get(self.components["rollouter"].get_elastic_statistics.remote())
            status["rollouter"] = elastic_stats
        except Exception:
            pass

        return status


# ============================================================================
# Helper: wrap existing trainer/rollouter for backward compatibility
# ============================================================================


def _add_rate_tracking_to_rollouter(rollouter):
    """
    Monkey-patch a rollouter to track total produced samples.

    This is used if the rollouter doesn't already have this method.
    """
    # Check if already has the method
    try:
        ray.get(rollouter.get_total_produced_samples.remote())
        return  # Already has it
    except Exception:
        pass

    # Add the method by setting an attribute
    # (Note: can't monkey-patch Ray actor, this is just documentation)
    logger.warning(
        "[ElasticScheduling] ElasticRollouter.get_total_produced_samples() not found. "
        "Make sure ElasticRollouter is used instead of FullyAsyncRollouter."
    )


def _add_rate_tracking_to_trainer(trainer):
    """
    Monkey-patch a trainer to track total consumed samples.

    This is used if the trainer doesn't already have this method.
    """
    try:
        ray.get(trainer.get_total_consumed_samples.remote())
        return  # Already has it
    except Exception:
        pass

    logger.warning(
        "[ElasticScheduling] ElasticTrainer.get_total_consumed_samples() not found. "
        "Make sure ElasticTrainer is used instead of FullyAsyncTrainer."
    )


# ============================================================================
# Main Entry Point
# ============================================================================


@hydra.main(config_path="config", config_name="elastic_ppo_trainer", version_base=None)
def main(config: DictConfig):
    """
    Main entry point for elastic scheduling PPO training.

    This function:
    1. Initializes Ray
    2. Creates ElasticSchedulingTaskRunner as a Ray actor
    3. Runs training to completion
    """
    logger.info(f"[ElasticScheduling] Starting with config:\n{OmegaConf.to_yaml(config)}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            address=getattr(config, "ray_address", "auto"),
            namespace=getattr(config, "ray_namespace", "verl_elastic"),
            ignore_reinit_error=True,
        )

    logger.info(f"[ElasticScheduling] Ray initialized. Resources: {ray.cluster_resources()}")

    # Create and run the task runner
    task_runner = ElasticSchedulingTaskRunner.remote()
    ray.get(task_runner.run.remote(config))

    logger.info("[ElasticScheduling] Training complete")


if __name__ == "__main__":
    main()
