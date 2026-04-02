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

    Elastic Resource Configuration
    ---------------------------------------------------------------
    Uses the same trainer config fields as standard training:
      trainer.nnodes          (int) : nodes in the elastic resource pool
      trainer.n_gpus_per_node (int) : GPUs per node in the elastic resource pool

    Total elastic GPU budget = trainer.nnodes × trainer.n_gpus_per_node.
    GPUs per elastic group   = actor_rollout_ref.actor.n_gpus_per_node (TP size).
    Number of elastic groups = total_elastic_gpus / gpus_per_group.

    No elastic resources when trainer.n_gpus_per_node == 0.
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
        # trainer_pool is only created when *actor* roles are requested.  In the elastic
        # scheduling scenario, actor GPUs come from elastic_worker_groups (not from a
        # static ResourcePool), so actor roles are intentionally excluded from the roles
        # list passed here.  Critic / RefPolicy roles alone must NOT trigger a trainer_pool
        # of size trainer.n_gpus_per_node (which represents the elastic GPU budget, not
        # the Critic's GPU requirement).
        actor_trigger_roles = [Role.Actor, Role.ActorRollout]
        trainer_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy, Role.RewardModel]
        trainer_n_gpus = int(getattr(config.trainer, "n_gpus_per_node", 0))
        trainer_nnodes = int(getattr(config.trainer, "nnodes", 0))
        if any(role in roles for role in actor_trigger_roles) and trainer_n_gpus > 0:
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
        # Resource layout:
        #   trainer.nnodes × trainer.n_gpus_per_node = total elastic GPU budget
        #
        # All elastic GPUs form a SINGLE RayWorkerGroup with a unified
        # PyTorch distributed process group (world_size = nnodes × n_gpus_per_node).
        # Inside this group Megatron/FSDP derives:
        #   DP = world_size / (TP × PP × CP)
        # and all DP replicas participate in gradient all-reduce simultaneously.
        #
        # Each elastic unit = TP×PP×CP ranks.  When switched to rollout, that
        # unit's ranks exit the DP group via rebuild_dp_group(new_active_ranks).
        # n_elastic_units = total_elastic_gpus / (TP × PP × CP).
        trainer_config = getattr(config, "trainer", {})
        trainer_nnodes = int(getattr(trainer_config, "nnodes", 0))
        trainer_n_gpus_per_node = int(getattr(trainer_config, "n_gpus_per_node", 0))
        total_elastic_gpus = trainer_nnodes * trainer_n_gpus_per_node

        if total_elastic_gpus == 0:
            self.components["elastic_worker_groups"] = []
            logger.info(
                f"[ElasticSchedulingTaskRunner] No elastic resources configured "
                f"(trainer.nnodes={trainer_nnodes}, trainer.n_gpus_per_node={trainer_n_gpus_per_node})"
            )
            return

        # Compute gpus_per_group = TP × PP × CP from actor megatron config.
        # This is the minimum switchable unit: when one elastic unit is flipped
        # to rollout, exactly gpus_per_group ranks leave the DP group.
        actor_config = getattr(config.actor_rollout_ref, "actor", {})
        megatron_config = getattr(actor_config, "megatron", None)
        if megatron_config is not None:
            tp = int(getattr(megatron_config, "tensor_model_parallel_size", 1))
            pp = int(getattr(megatron_config, "pipeline_model_parallel_size", 1))
            cp = int(getattr(megatron_config, "context_parallel_size", 1))
            gpus_per_unit = tp * pp * cp
        else:
            gpus_per_unit = 1

        if total_elastic_gpus % gpus_per_unit != 0:
            raise ValueError(
                f"total_elastic_gpus ({total_elastic_gpus}) must be divisible by "
                f"gpus_per_unit=TP×PP×CP ({gpus_per_unit})"
            )
        n_elastic_units = total_elastic_gpus // gpus_per_unit

        logger.info(
            f"[ElasticSchedulingTaskRunner] Elastic resource budget: "
            f"{trainer_nnodes}×{trainer_n_gpus_per_node}={total_elastic_gpus} GPUs, "
            f"gpus_per_unit(TP×PP×CP)={gpus_per_unit}, "
            f"n_elastic_units={n_elastic_units} (initial DP={n_elastic_units})"
        )

        from verl.experimental.elastic_scheduling.elastic_engine_workers import ElasticActorWorker
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

        ray_worker_group_cls = self.components.get("ray_worker_group_cls", RayWorkerGroup)

        # Create ONE large RayWorkerGroup covering all elastic GPUs.
        # process_on_nodes = [n_gpus_per_node] * nnodes ensures all workers
        # share a single dist.init_process_group world (world_size = total_elastic_gpus).
        resource_pool = RayResourcePool(
            process_on_nodes=[trainer_n_gpus_per_node] * trainer_nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="elastic_actor",
        )
        ray_cls_with_init = RayClassWithInitArgs(
            cls=ray.remote(ElasticActorWorker),
            config=config.actor_rollout_ref,
            role="actor_rollout",
        )
        wg = ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix="elastic_actor",
        )
        ray.get(wg.execute_all("init_model"))
        logger.info(
            f"[ElasticSchedulingTaskRunner] Elastic worker group created "
            f"({trainer_nnodes}×{trainer_n_gpus_per_node} GPUs, world_size={total_elastic_gpus})"
        )

        # Map each elastic unit to its contiguous rank range within the wg.
        # elastic_0 → [0 .. gpus_per_unit-1]
        # elastic_1 → [gpus_per_unit .. 2*gpus_per_unit-1]  etc.
        elastic_unit_ranks: dict[str, list[int]] = {}
        for i in range(n_elastic_units):
            unit_id = f"elastic_{i}"
            start = i * gpus_per_unit
            elastic_unit_ranks[unit_id] = list(range(start, start + gpus_per_unit))

        # Store: (resource_id, wg) list for backward compatibility with _wire / _build helpers
        # All entries point to the SAME wg; resource_id is just a logical unit label.
        elastic_worker_groups = [(uid, wg) for uid in elastic_unit_ranks]
        self.components["elastic_worker_groups"] = elastic_worker_groups
        self.components["elastic_wg"] = wg
        self.components["elastic_unit_ranks"] = elastic_unit_ranks
        logger.info(
            f"[ElasticSchedulingTaskRunner] {n_elastic_units} elastic unit(s) defined "
            f"(all share the same RayWorkerGroup, world_size={total_elastic_gpus})"
        )

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

        # Elastic actor GPUs are managed by elastic_worker_groups (created separately in
        # _create_elastic_worker_groups), NOT through ResourcePoolManager.  Pass only
        # non-actor roles (Critic, RefPolicy, RewardModel) to create_rp so that no
        # trainer_pool is created for actor/actor_rollout roles.
        actor_roles = {Role.Actor, Role.ActorRollout}
        non_actor_roles = [r for r in trainer_roles if r not in actor_roles]
        trainer_resource_pool_manager = create_rp(config, roles=non_actor_roles) if create_rp else None

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

        # All elastic units share the same large RayWorkerGroup; inject it once.
        elastic_wg = self.components.get("elastic_wg")
        if elastic_wg is not None:
            ray.get(rollouter.set_elastic_worker_group.remote(elastic_wg))
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
        Register the single large elastic RayWorkerGroup plus the per-unit rank
        mapping with ElasticTrainer so that switch_elastic_to_train/rollout()
        can look up which ranks to add/remove from the DP group.

        Also wire rollouter reference into trainer (needed for the trainer to
        call add/remove_elastic_replica on the rollouter during a switch).
        """
        trainer = self.components["trainer"]
        elastic_wg = self.components.get("elastic_wg")
        elastic_unit_ranks: dict = self.components.get("elastic_unit_ranks", {})

        if elastic_wg is not None:
            # Register the unified wg once, then register the rank mapping for
            # each logical elastic unit.
            ray.get(trainer.register_elastic_worker_group.remote("elastic_wg", elastic_wg))
            ray.get(trainer.register_elastic_unit_ranks.remote(elastic_unit_ranks))
            logger.info(
                f"[ElasticSchedulingTaskRunner] Registered unified elastic wg + "
                f"{len(elastic_unit_ranks)} unit rank mapping(s) with ElasticTrainer"
            )

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
            # [DEBUG] Force Train→Rollout switch every step for system sanity-checking
            "debug_force_switch_every_step": bool(getattr(elastic_config, "debug_force_switch_every_step", False)),
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
                    # ElasticTrainer registers all units in TRAIN mode initially
                    # (see register_elastic_unit_ranks → _elastic_active_units = set(unit_ranks.keys())).
                    # Coordinator must mirror that state so it can find candidates
                    # when the first scale_rollout action fires.
                    "initial_mode": "train",
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

    # Mirror rollout topology into actor_rollout_ref.rollout so that
    # FullyAsyncAgentLoopManager can infer world size in standalone mode.
    # (Same as fully_async_main.py does before launching the task runner.)
    OmegaConf.update(config, "actor_rollout_ref.rollout.nnodes", config.rollout.nnodes, merge=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.n_gpus_per_node", config.rollout.n_gpus_per_node, merge=True)

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
