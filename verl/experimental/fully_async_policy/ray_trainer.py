# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.rollout_skip import RolloutSkip


class FullyAsyncRayPPOTrainer(RayPPOTrainer):
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_async_rollout_manager()

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    def _create_worker_classes(self):
        self._create_actor_rollout_classes()
        self._create_critic_class()
        self._create_reference_policy_class()
        self._create_reward_model_class()

    def _create_actor_rollout_classes(self):
        raise NotImplementedError

    def _create_critic_class(self):
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    def _create_reference_policy_class(self):
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
                # profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    def _create_reward_model_class(self):
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    def _init_worker_groups(self):
        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        self.all_wg = all_wg

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = self.all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

    def _init_async_rollout_manager(self):
        pass


    def _prepare_generate_batch(self, batch_dict):
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        return batch, gen_batch

    def _post_generate_batch(self, batch, gen_batch_output, metrics):
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        # TODO: Decouple the DP balancing and mini-batching.
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return batch

    def _process_batch_common(self, batch, metrics, timing_raw, local_trigger_step=None):
        with marked_timer("reward", timing_raw, color="yellow"):
            # compute reward model score
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        with marked_timer("old_log_prob", timing_raw, color="blue"):

            def compute_old_log_prob(batch):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                actor_config = self.config.actor_rollout_ref.actor
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=response_masks,
                    loss_agg_mode=actor_config.loss_agg_mode,
                    loss_scale_factor=actor_config.loss_scale_factor,
                )
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)
                if "rollout_log_probs" in batch.batch.keys():
                    # TODO: we may want to add diff of probs too.
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))
                return batch

            async_training = self.config.get("async_training", None)
            if async_training and async_training.use_rollout_log_probs:
                # If local_triger_step == 1, load the training engine's parameters to the CPU
                #  and save a copy for subsequent MIS use.
                # If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
                # then restore the parameters of the current version.
                if local_trigger_step == 1:
                    self.actor_rollout_wg.save_model_to_cpu(1)
                    batch = compute_old_log_prob(batch)
                elif local_trigger_step is not None:
                    self.actor_rollout_wg.save_model_to_cpu(local_trigger_step)
                    self.actor_rollout_wg.restore_model_from_cpu(1)
                    batch = compute_old_log_prob(batch)
                    self.actor_rollout_wg.restore_model_from_cpu(local_trigger_step)
                    self.actor_rollout_wg.clear_cpu_model(local_trigger_step)
                else:
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            else:
                batch = compute_old_log_prob(batch)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer("adv", timing_raw, color="brown"):
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # Compute rollout correction weights centrally (once per batch)
            # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
            # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                # IS and off-policy metrics already have rollout_corr/ prefix
                metrics.update(is_metrics)

            # compute advantages, executed on the driver process
            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return batch, reward_extra_infos_dict

    def _log_rollout(self, batch, reward_extra_infos_dict, timing_raw):
        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )

                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    gts=sample_gts,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

    async def _validate_metrics(self, is_last_step, last_val_metrics, metrics, timing_raw):
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("testing", timing_raw, color="green"):
                val_metrics: dict = await self._validate()
                if is_last_step:
                    last_val_metrics = val_metrics
            metrics.update(val_metrics)
            return last_val_metrics

    def _collect_metrics(self, batch, epoch, metrics, timing_raw):
        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)

        # training metrics
        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
            }
        )
        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

    def _post_batch_processing(self, batch: DataProto):
        # this is experimental and may be changed/removed in the future in favor of a general-purpose one
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)

        # this is experimental and may be changed/removed in the future
        # in favor of a general-purpose data buffer pool
        if hasattr(self.train_dataset, "on_batch_end"):
            # The dataset may be changed after each training batch
            self.train_dataset.on_batch_end(batch=batch)

    async def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                raise RuntimeError("self.async_rollout_mode is False not supported.")
            else:
                test_output_gen_batch_padded = await self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

