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

Wires together all elastic scheduling components.

Component diagram
-----------------

    ElasticSchedulingTaskRunner (Ray actor, num_cpus=4)
    │
    ├── ElasticRollouter (Ray actor)
    │     └── ElasticAgentLoopManager
    │             ├── Fixed rollout replicas  (dedicated rollout GPU pool)
    │             └── Elastic hybrid replicas (shared GPU pool, created at init, sleeping)
    │
    ├── ElasticTrainer (Ray actor)
    │     ├── Fixed actor worker group  (dedicated training GPUs)
    │     └── Elastic actor registry    (elastic wgs added/removed via switch API)
    │
    ├── MessageQueue (Ray actor)  — FIFO sample buffer
    │
    └── ElasticCoordinator (Ray actor)
            ├── Polls queue metrics  →  decides pending_action
            └── on_before_fit_step() hook (called by ElasticTrainer)
                  └── delegates switch to ElasticTrainer.switch_elastic_to_rollout/train()

Elastic resource lifecycle
--------------------------
Elastic resources are ActorRolloutRef worker groups whose GPUs are **shared**
between the training engine and rollout servers.

1. At init time:
   a. ElasticActorWorkerGroup is created (same process as training).
   b. Each worker calls init_model() → builds actor engine.
   c. ElasticRollouter.set_elastic_worker_group(elastic_wg) is called.
   d. ElasticRollouter.init_workers() uses elastic_wg to call
      ElasticAgentLoopManager.create(elastic_worker_group=elastic_wg), which
      creates RolloutReplica objects and puts them to sleep immediately.
   e. ElasticTrainer.register_elastic_worker_group(resource_id, wg) is called
      for each elastic wg so that switch_elastic_to_train() can find them.

2. During training:
   TRAIN → ROLLOUT (triggered by ElasticCoordinator via ElasticTrainer):
     a. ElasticTrainer.remove_elastic_actor()  – DP rebuild without this wg
     b. elastic_wg.switch_to_rollout()          – offload actor to CPU
     c. ElasticRollouter.add_elastic_replica()  – wake up rollout server

   ROLLOUT → TRAIN (triggered by ElasticCoordinator via ElasticTrainer):
     a. ElasticRollouter.remove_elastic_replica() – sleep rollout server
     b. elastic_wg.switch_to_train()               – load actor to GPU
     c. ElasticTrainer.add_elastic_actor()          – DP rebuild with this wg

Usage
-----
    python -m verl.experimental.elastic_scheduling.main \\
        --config-path config --config-name elastic_ppo_trainer \\
        trainer.nnodes=2 trainer.n_gpus_per_node=8 \\
        rollout.nnodes=1 rollout.n_gpus_per_node=8 \\
        elastic_scheduling.n_elastic_resources=1 \\
        elastic_scheduling.elastic_n_gpus_per_node=4
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
    1. ElasticRollouter : async sample generation with dynamic replica pool
    2. ElasticTrainer   : async training with dynamic DP pool
    3. MessageQueue     : FIFO buffer between rollouter and trainer
    4. ElasticCoordinator: monitoring + role switching loop

    Elastic Resource Configuration (from config.elastic_scheduling)
    ---------------------------------------------------------------
    n_elastic_resources       (int)    : number of elastic resource groups
    elastic_n_gpus_per_node   (int)    : GPUs per node for each elastic group
    elastic_n_nodes           (int)    : nodes per elastic group  (default 1)
    elastic_initial_mode      (str)    : "rollout" | "train" (default "rollout")
    rollout_queue_high_watermark (float): scale-rollout threshold (default 0.8)
    rollout_queue_low_watermark  (float): scale-train threshold   (default 0.3)
    cooldown_seconds          (float)  : min seconds between switches (default 30)
    check_interval            (float)  : monitoring poll period (default 2)
    """

    def __init__(self):
        self.running = False
        self.components: dict = {}
        self._coordinator_task: Optional[asyncio.Task] = None

    def run(self, config: DictConfig):
        """Main entry point. Initializes all components and starts training."""
        logger.info(f"[ElasticSchedulingTaskRunner] Starting on {socket.gethostname()}")
        self._initialize_components(config)
        self._run_training_loop()

    # =========================================================================
    # Component Initialization
    # =========================================================================

    def _initialize_components(self, config: DictConfig):
        """Initialize all components in the correct dependency order."""
        logger.info("[ElasticSchedulingTaskRunner] Initializing components...")

        # 1. Load tokenizer / processor
        self._init_tokenizer(config)

        # 2. Create role-worker mappings
        self._init_resource_mappings(config)

        # 3. Create elastic worker groups (BEFORE rollouter/trainer init)
        self._init_elastic_worker_groups(config)

        # 4. Create ElasticTrainer (reserves fixed training GPUs)
        self._create_elastic_trainer(config)

        # 5. Create ElasticRollouter (injects elastic wg before init_workers)
        self._create_elastic_rollouter(config)

        # 6. Wire elastic wg references between trainer / rollouter
        self._wire_elastic_worker_groups(config)

        # 7. Setup MessageQueue and connect to rollouter + trainer
        self._init_message_queue(config)

        # 8. Setup ElasticCoordinator
        self._init_coordinator(config)

        # 9. Connect rollouter → trainer (initializes checkpoint manager)
        self._init_elastic_param_sync(config)

        # 10. Load checkpoints
        self._load_checkpoints()

        # 11. Initial parameter sync
        self._initial_param_sync()

        logger.info("[ElasticSchedulingTaskRunner] All components initialized")

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------

    def _init_tokenizer(self, config: DictConfig):
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        self.components["tokenizer"] = hf_tokenizer(local_path)
        self.components["processor"] = hf_processor(local_path, use_fast=False)
        self.components["config"] = config
        logger.info("[ElasticSchedulingTaskRunner] Tokenizer loaded")

    # -------------------------------------------------------------------------
    # Role / Resource Mappings
    # -------------------------------------------------------------------------

    def _create_resource_pool_manager(self, config: DictConfig, roles: list):
        """
        Create a ResourcePoolManager for the given roles.

        Mirrors the pattern in verl.experimental.separation.utils but lives here
        so that elastic_scheduling.main has no hard dependency on the separation
        sub-package.

        Two resource pools are supported:
        - ``trainer_pool``: covers Actor / ActorRollout / Critic / RefPolicy /
          RewardModel roles.  Sized from ``config.trainer.{nnodes,n_gpus_per_node}``.
        - ``rollout_pool``: covers the standalone Rollout role.  Sized from
          ``config.rollout.{nnodes,n_gpus_per_node}``.

        Args:
            config: Top-level Hydra/OmegaConf config.
            roles: List of :class:`~verl.trainer.ppo.utils.Role` values that
                need resource pools.

        Returns:
            ResourcePoolManager
        """
        from verl.single_controller.ray import ResourcePoolManager
        from verl.trainer.ppo.utils import Role

        resource_pool_spec: dict[str, list[int]] = {}
        mapping: dict = {}

        # --- trainer pool (Actor / ActorRollout / Critic / RefPolicy / RewardModel) ---
        # In fully-elastic mode, all training GPUs come from elastic worker groups so
        # trainer.n_gpus_per_node may legitimately be 0.  Skip fixed trainer pool in
        # that case – the elastic worker groups are wired in separately.
        trainer_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy, Role.RewardModel]
        trainer_n_gpus = int(getattr(config.trainer, "n_gpus_per_node", 0))
        trainer_nnodes = int(getattr(config.trainer, "nnodes", 0))
        if any(role in roles for role in trainer_roles) and trainer_n_gpus > 0:
            assert trainer_nnodes > 0, "config.trainer.nnodes must be > 0"

            trainer_pool = [trainer_n_gpus] * trainer_nnodes
            resource_pool_spec["trainer_pool"] = trainer_pool

            for role in trainer_roles:
                if role in roles:
                    mapping[role] = "trainer_pool"

        # --- rollout pool (standalone Rollout role) ---
        if Role.Rollout in roles:
            assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be > 0"
            assert config.rollout.nnodes > 0, "config.rollout.nnodes must be > 0"

            rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
            resource_pool_spec["rollout_pool"] = rollout_pool
            mapping[Role.Rollout] = "rollout_pool"

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    def _create_role_worker_mapping(self, config: DictConfig):
        """
        Build the role → worker-class mapping for ElasticTrainer / ElasticRollouter.

        Mirrors the pattern in verl.experimental.separation.utils.create_role_worker_mapping
        but lives here to keep elastic_scheduling self-contained.

        Supported worker implementations:
        - ``DetachActorWorker`` (from separation.engine_workers) for Actor /
          ActorRollout / RefPolicy roles.
        - ``TrainingWorker`` (from workers.engine_workers) for Critic.

        The function requires ``config.trainer.use_legacy_worker_impl == "disable"``.

        Args:
            config: Top-level Hydra/OmegaConf config.

        Returns:
            tuple[dict, type[RayWorkerGroup]]:
                role_worker_mapping, ray_worker_group_cls
        """
        import ray as _ray

        from verl.experimental.separation.engine_workers import DetachActorWorker
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.utils import Role, need_reference_policy
        from verl.workers.engine_workers import TrainingWorker

        if config.trainer.get("use_legacy_worker_impl", "auto") != "disable":
            raise NotImplementedError(
                "ElasticScheduling requires use_legacy_worker_impl=disable (new engine worker implementation)"
            )

        ray_worker_group_cls = RayWorkerGroup

        # Must mirror the logic in FullyAsyncTrainer.__init__ so that the key in
        # role_worker_mapping matches self.train_role inside the trainer.
        use_trainer_do_validate = bool(config.get("async_training", {}).get("use_trainer_do_validate", False))
        train_role = Role.ActorRollout if use_trainer_do_validate else Role.Actor

        role_worker_mapping = {
            train_role: _ray.remote(DetachActorWorker),
            Role.Critic: _ray.remote(TrainingWorker),
        }

        # Add reference policy when KL-loss / reference reward is required
        if need_reference_policy(config):
            role_worker_mapping[Role.RefPolicy] = _ray.remote(DetachActorWorker)

        return role_worker_mapping, ray_worker_group_cls

    def _init_resource_mappings(self, config: DictConfig):
        """
        Populate ``self.components`` with:
        - ``role_worker_mapping``         – role → Ray remote worker class
        - ``ray_worker_group_cls``        – worker-group class (RayWorkerGroup)
        - ``create_resource_pool_manager``– callable(config, roles) → ResourcePoolManager

        Uses the local helpers ``_create_role_worker_mapping`` and
        ``_create_resource_pool_manager`` so that this module has no hard
        dependency on ``verl.experimental.separation.utils``.
        """
        role_worker_mapping, ray_worker_group_cls = self._create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls
        # Store as a bound-method so callers can do create_rp(config, roles=...)
        self.components["create_resource_pool_manager"] = self._create_resource_pool_manager

    # -------------------------------------------------------------------------
    # Elastic Worker Groups
    # -------------------------------------------------------------------------

    def _init_elastic_worker_groups(self, config: DictConfig):
        """
        Create one ElasticActorWorker group per elastic resource entry.

        Each group is an ActorRolloutRef worker group whose GPUs are shared
        between the training engine and the rollout server.  After creation:
        - Workers call init_model() to initialise the actor engine.
        - Each group is stored as components["elastic_wg_{i}"].
        - A combined list is kept as components["elastic_worker_groups"].

        The worker groups are NOT yet connected to the rollouter / trainer;
        that happens in _wire_elastic_worker_groups().
        """
        elastic_config = getattr(config, "elastic_scheduling", {})
        n_elastic = int(getattr(elastic_config, "n_elastic_resources", 0))

        if n_elastic == 0:
            self.components["elastic_worker_groups"] = []
            logger.info("[ElasticSchedulingTaskRunner] No elastic resources configured")
            return

        from verl.experimental.elastic_scheduling.elastic_engine_workers import ElasticActorWorker
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

        elastic_n_gpus_per_node = int(getattr(elastic_config, "elastic_n_gpus_per_node", 1))
        elastic_n_nodes = int(getattr(elastic_config, "elastic_n_nodes", 1))
        ray_worker_group_cls = self.components.get("ray_worker_group_cls", RayWorkerGroup)

        elastic_worker_groups = []
        for i in range(n_elastic):
            resource_id = f"elastic_{i}"
            try:
                resource_pool = RayResourcePool(
                    process_on_nodes=[elastic_n_gpus_per_node] * elastic_n_nodes,
                    use_gpu=True,
                    max_colocate_count=1,
                    name_prefix=f"elastic_actor_{i}",
                )
                ray_cls_with_init = RayClassWithInitArgs(
                    cls=ray.remote(ElasticActorWorker),
                    config=config.actor_rollout_ref,
                    role="actor_rollout",
                )
                wg = ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=ray_cls_with_init,
                    name_prefix=f"elastic_actor_{i}",
                )
                # Initialise actor model on the workers
                ray.get(wg.execute_all("init_model"))
                elastic_worker_groups.append((resource_id, wg))
                self.components[f"elastic_wg_{i}"] = wg
                logger.info(
                    f"[ElasticSchedulingTaskRunner] Elastic worker group '{resource_id}' created "
                    f"({elastic_n_nodes}×{elastic_n_gpus_per_node} GPUs)"
                )
            except Exception as e:
                logger.error(f"[ElasticSchedulingTaskRunner] Failed to create elastic wg {i}: {e}")
                raise

        self.components["elastic_worker_groups"] = elastic_worker_groups
        logger.info(f"[ElasticSchedulingTaskRunner] {n_elastic} elastic worker group(s) created and model-initialised")

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------

    def _create_elastic_trainer(self, config: DictConfig):
        from verl.experimental.elastic_scheduling.elastic_trainer import ElasticTrainer
        from verl.trainer.ppo.utils import Role

        create_rp = self.components.get("create_resource_pool_manager")
        role_worker_mapping = self.components.get("role_worker_mapping", {})
        ray_worker_group_cls = self.components.get("ray_worker_group_cls")

        trainer_roles = {role: wcls for role, wcls in role_worker_mapping.items() if role != Role.Rollout}
        trainer_resource_pool_manager = create_rp(config, roles=list(trainer_roles.keys())) if create_rp else None

        trainer = (
            ray.remote(ElasticTrainer)
            .options(num_cpus=10, max_concurrency=100)
            .remote(
                config=config,
                tokenizer=self.components["tokenizer"],
                role_worker_mapping=trainer_roles,
                resource_pool_manager=trainer_resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                processor=self.components["processor"],
            )
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        logger.info("[ElasticSchedulingTaskRunner] ElasticTrainer created and workers initialized")

        # -------------------------------------------------------------------------
        # Rollouter (injects elastic wg before init_workers)
        # -------------------------------------------------------------------------

    def _create_elastic_rollouter(self, config: DictConfig):
        """
        Create ElasticRollouter and inject the elastic worker group BEFORE
        calling init_workers() so that ElasticAgentLoopManager.create()
        receives elastic_worker_group and initialises hybrid replicas.
        """
        from verl.experimental.elastic_scheduling.elastic_rollouter import ElasticRollouter
        from verl.trainer.ppo.utils import Role

        create_rp = self.components.get("create_resource_pool_manager")
        role_worker_mapping = self.components.get("role_worker_mapping", {})
        ray_worker_group_cls = self.components.get("ray_worker_group_cls")

        rollouter_resource_pool_manager = create_rp(config, roles=[Role.Rollout]) if create_rp else None

        rollouter = (
            ray.remote(ElasticRollouter)
            .options(num_cpus=10, max_concurrency=100)
            .remote(
                config=config,
                tokenizer=self.components["tokenizer"],
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=rollouter_resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                processor=self.components["processor"],
            )
        )

        # Inject the combined elastic worker group (all groups merged or first group,
        # depending on whether there is exactly one elastic DP slice at a time).
        elastic_worker_groups = self.components.get("elastic_worker_groups", [])
        if elastic_worker_groups:
            # For now inject only the first elastic wg.
            # TODO: support multiple elastic replicas by supplying a merged wg.
            _, first_wg = elastic_worker_groups[0]
            ray.get(rollouter.set_elastic_worker_group.remote(first_wg))
            logger.info("[ElasticSchedulingTaskRunner] Injected elastic worker group into rollouter")

        ray.get(rollouter.init_workers.remote())
        # Note: set_max_required_samples requires message_queue_client set first
        self.components["rollouter"] = rollouter
        logger.info("[ElasticSchedulingTaskRunner] ElasticRollouter created and workers initialized")

    # -------------------------------------------------------------------------
    # Wire elastic worker groups into trainer
    # -------------------------------------------------------------------------

    def _wire_elastic_worker_groups(self, config: DictConfig):
        """
        Register elastic worker groups with ElasticTrainer so that
        switch_elastic_to_train() can look them up at switch time.

        Also wire rollouter reference into trainer (needed for the trainer to
        call add/remove_elastic_replica on the rollouter during a switch).
        """
        trainer = self.components["trainer"]
        elastic_worker_groups = self.components.get("elastic_worker_groups", [])
        for resource_id, wg in elastic_worker_groups:
            ray.get(trainer.register_elastic_worker_group.remote(resource_id, wg))
            logger.info(f"[ElasticSchedulingTaskRunner] Registered '{resource_id}' with ElasticTrainer")

        # Wire rollouter into trainer for switch sequences
        ray.get(trainer.set_rollouter.remote(self.components["rollouter"]))
        logger.info("[ElasticSchedulingTaskRunner] Rollouter wired into ElasticTrainer")

        # Register hybrid replicas with the trainer's checkpoint manager
        # (need to get replica objects from the rollouter after init_workers)
        try:
            all_replicas = ray.get(self.components["rollouter"].get_all_elastic_replicas.remote())
            if all_replicas:
                ray.get(trainer.register_hybrid_replicas.remote(all_replicas))
                logger.info(
                    f"[ElasticSchedulingTaskRunner] Registered {len(all_replicas)} hybrid replica(s) "
                    "with ElasticCheckpointManager"
                )
        except Exception as e:
            logger.warning(f"[ElasticSchedulingTaskRunner] Could not register hybrid replicas: {e}")

    # -------------------------------------------------------------------------
    # MessageQueue
    # -------------------------------------------------------------------------

    def _init_message_queue(self, config: DictConfig):
        from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient

        ray.get(self.components["rollouter"].set_max_required_samples.remote())
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())

        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))
        logger.info(
            f"[ElasticSchedulingTaskRunner] total_train_steps={total_train_steps}, max_queue_size={max_queue_size}"
        )

        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))
        logger.info(f"[ElasticSchedulingTaskRunner] MessageQueue initialized (capacity={max_queue_size})")

    # -------------------------------------------------------------------------
    # Coordinator
    # -------------------------------------------------------------------------

    def _init_coordinator(self, config: DictConfig):
        from verl.experimental.elastic_scheduling.coordinator import ElasticCoordinator

        elastic_config = getattr(config, "elastic_scheduling", {})

        # Build resource info from the elastic_worker_groups we created
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

        # Wire coordinator into trainer for pre-fit-step hook
        ray.get(self.components["trainer"].set_elastic_coordinator.remote(coordinator))

        logger.info(
            f"[ElasticSchedulingTaskRunner] ElasticCoordinator initialized with "
            f"{len(elastic_resource_infos)} elastic resource(s)"
        )

    def _build_elastic_resource_infos(self, config: DictConfig) -> list[dict]:
        """
        Build the ElasticResourceInfo list from already-created elastic worker groups.

        Each entry maps a resource_id to its initial mode and the worker handles
        from the corresponding elastic worker group.  The coordinator uses this
        to track state and (optionally) issue direct worker calls for monitoring.

        The actual role-switch operations are always delegated to ElasticTrainer;
        worker_handles here are for state tracking only.
        """
        elastic_config = getattr(config, "elastic_scheduling", {})
        initial_mode = getattr(elastic_config, "elastic_initial_mode", "rollout")

        elastic_worker_groups = self.components.get("elastic_worker_groups", [])
        elastic_resource_infos = []

        for resource_id, wg in elastic_worker_groups:
            # Retrieve global ranks from the worker group as proxy for "handles"
            try:
                worker_handles = wg.get_all_actor_handles()
            except AttributeError:
                # RayWorkerGroup may not have this helper; fall back to empty list.
                worker_handles = []

            elastic_resource_infos.append(
                {
                    "resource_id": resource_id,
                    "initial_mode": initial_mode,
                    "worker_handles": worker_handles,
                }
            )

        return elastic_resource_infos

    # -------------------------------------------------------------------------
    # Parameter Sync
    # -------------------------------------------------------------------------

    def _init_elastic_param_sync(self, config: DictConfig):
        """
        Wire rollouter into trainer.  This triggers ElasticTrainer to call
        _setup_checkpoint_manager() which creates an ElasticCheckpointManager
        covering both fixed (standalone) and elastic (hybrid) replicas.
        """
        # Note: rollouter was already wired in _wire_elastic_worker_groups;
        # call it again only if set_rollouter is idempotent.
        # The checkpoint manager is initialised inside set_rollouter().
        logger.info("[ElasticSchedulingTaskRunner] Elastic parameter sync ready (via ElasticCheckpointManager)")

    # -------------------------------------------------------------------------
    # Checkpoint + Initial Sync
    # -------------------------------------------------------------------------

    def _load_checkpoints(self):
        logger.info("[ElasticSchedulingTaskRunner] Loading checkpoints...")
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

    def _initial_param_sync(self):
        logger.info("[ElasticSchedulingTaskRunner] Performing initial parameter sync...")
        try:
            ray.get(self.components["trainer"]._fit_update_weights.remote())
            logger.info("[ElasticSchedulingTaskRunner] Initial parameter sync complete")
        except Exception as e:
            logger.warning(
                f"[ElasticSchedulingTaskRunner] Initial param sync failed "
                f"(may be OK if checkpoint was already loaded): {e}"
            )

    # =========================================================================
    # Training Loop
    # =========================================================================

    def _run_training_loop(self):
        self.running = True
        logger.info("[ElasticSchedulingTaskRunner] Starting training loop...")

        # Start coordinator background monitoring
        self.components["coordinator"].start.remote()

        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        try:
            futures = [rollouter_future, trainer_future]
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)
                for future in done_futures:
                    try:
                        ray.get(future)
                        logger.info("[ElasticSchedulingTaskRunner] Component completed successfully")
                    except Exception as e:
                        logger.error(f"[ElasticSchedulingTaskRunner] Component failed: {e}")
                        for remaining in remaining_futures:
                            ray.cancel(remaining)
                        raise
                futures = remaining_futures

        except Exception as e:
            logger.error(f"[ElasticSchedulingTaskRunner] Training failed: {e}")
            raise

        finally:
            logger.info("[ElasticSchedulingTaskRunner] Stopping coordinator...")
            ray.get(self.components["coordinator"].stop.remote())

            message_queue_client = self.components.get("message_queue_client")
            if message_queue_client:
                asyncio.run(message_queue_client.clear_queue())

            self.running = False
            logger.info("[ElasticSchedulingTaskRunner] Training completed")

    # =========================================================================
    # Monitoring
    # =========================================================================

    def get_status(self) -> dict:
        status = {"running": self.running}

        for key, method in [
            ("coordinator", "get_status"),
            ("trainer", "get_elastic_statistics"),
            ("rollouter", "get_elastic_statistics"),
        ]:
            try:
                component = self.components.get(key)
                if component is not None:
                    status[key] = ray.get(getattr(component, method).remote())
            except Exception:
                pass

        return status


# ============================================================================
# Main Entry Point
# ============================================================================


@hydra.main(config_path="config", config_name="elastic_ppo_trainer", version_base=None)
def main(config: DictConfig):
    """
    Main entry point for elastic scheduling PPO training.

    1. Initialize Ray.
    2. Create ElasticSchedulingTaskRunner as a Ray actor.
    3. Run training to completion.
    """
    logger.info(f"[ElasticScheduling] Starting with config:\n{OmegaConf.to_yaml(config)}")

    if not ray.is_initialized():
        ray.init(
            address=getattr(config, "ray_address", "auto"),
            namespace=getattr(config, "ray_namespace", "verl_elastic"),
            ignore_reinit_error=True,
        )

    logger.info(f"[ElasticScheduling] Ray initialized. Resources: {ray.cluster_resources()}")

    task_runner = ElasticSchedulingTaskRunner.remote()
    ray.get(task_runner.run.remote(config))

    logger.info("[ElasticScheduling] Training complete")


if __name__ == "__main__":
    main()
