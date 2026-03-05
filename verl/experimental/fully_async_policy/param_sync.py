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

import logging
import time

import ray

from verl.checkpoint_engine import CheckpointEngineManager
from verl.utils.config import omega_conf_to_dataclass

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer, responsible for synchronizing model parameters between actor and rollout
    Based on the mature synchronization mode implementation of one_step_off_policy
    Merges the functions of the original multiple synchronizer classes
    """

    def __init__(self, config, trainer, rollouter, mq):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter
        self.mq_client = mq
        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        # Basic attributes
        self.wait_last_update = None
        self.wait_last_resume = None
        self.validate_task = None

        # Statistics
        self.current_version = 0

        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )

    def sync_weights(self, version, validate=False, global_steps=0, use_trainer_do_validate=False):
        """Sync weights between trainer and rollouter, and update parameter version"""
        start_time = time.time()
        self.current_version = version
        self.checkpoint_manager.update_weights(global_steps)
        end_time = time.time()
        print(f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f} seconds")
        # async train do validate
        print(f"[ParameterSynchronizer] validate: {validate}, use_trainer_do_validate: {use_trainer_do_validate}")
        if validate and use_trainer_do_validate:
            print("[ParameterSynchronizer] use trainer to do validate")
            self.validate_task = self.trainer._validate_process.remote()
        else:
            self.validate_task = None
        # Async Update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(
            version, validate, global_steps, use_trainer_do_validate
        )

    def wait_last_valid(self):
        print("[ParameterSynchronizer] Waiting last sync and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.validate_task:
            ray.get(self.validate_task)
        print(f"[ParameterSynchronizer] Wait last validate cost: {time.time() - start_time:.2f} seconds")

    def rollouter_save_checkpoint(self, local_global_step_folder: str):
        """Trigger rollout to save checkpoint(dataloader)"""
        print(f"[ParameterSynchronizer] Triggering checkpoint save at {local_global_step_folder} ...")
        return ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
