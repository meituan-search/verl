# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Partial Re-prefill trainer: per-trajectory case dispatch.

Extends PPOTrainerColocateAsync. Sibling to reprefill_decoupled.

Three cases per trajectory at sample time:
- Case 1 (piggyback): client already populated new_rollout_log_probs during
  partial_rollout resume. Skip.
- Case 2 (fully-stale): full reprefill at W_sample. Existing _reprefill_all logic.
- Case 3 (fully-fresh): all tokens at current weight. Copy rollout_log_probs.

Enable via: trainer.v1.trainer_mode=partial_reprefill
"""

import logging
import os

import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.reprefill_utils import (
    build_reprefill_inputs,
    decide_case,
    reprefill_trajectories,
    slice_response_logprobs,
    to_nested_jagged,
)
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.utils.debug import marked_timer
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("partial_reprefill")
class PPOTrainerPartialReprefill(PPOTrainerColocateAsync):
    """Partial reprefill trainer (colocate async)."""

    def on_train_begin(self):
        cfg = self.config.trainer.v1.partial_reprefill
        num_warmup_batches = cfg.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches (partial_reprefill)")

    def on_sampled(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
            batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    @auto_await
    async def _reprefill_all(self, prompt_ids_list):
        return await reprefill_trajectories(
            client=self.get_llm_client(),
            prompt_ids_list=prompt_ids_list,
            request_prefix=f"partial_reprefill_{self.global_steps}",
        )

    def _compute_new_rollout_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        cfg = self.config.trainer.v1.partial_reprefill
        current_version = self.global_steps - 1
        resume_version = current_version

        # Pre-fetch rollout_log_probs + token_versions for all keys (case 3 needs
        # rollout_log_probs; case dispatch needs token_versions[-1]).
        fields = ["rollout_log_probs", "token_versions"]
        meta = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )

        # Determine which trajectories need full reprefill (case 2).
        case2_indices: list[int] = []
        case3_indices: list[int] = []
        case1_count = 0
        for i, tag in enumerate(batch.tags):
            tag = tag or {}
            piggyback = bool(tag.get("piggyback_marker", False)) and cfg.enable_piggyback
            # token_versions is a nested-jagged int32 tensor; last element of trajectory i.
            tv = meta["token_versions"][i] if "token_versions" in meta else None
            last_tv = int(tv[-1].item()) if tv is not None and len(tv) > 0 else None
            case = decide_case(
                piggyback_marker=piggyback,
                last_token_version=last_tv,
                current_parameter_version=current_version,
                enable_case_skip=cfg.enable_case_skip,
                enable_piggyback=cfg.enable_piggyback,
            )
            if case == 1:
                case1_count += 1
            elif case == 3:
                case3_indices.append(i)
            else:
                case2_indices.append(i)

        # Case 2: full reprefill for selected keys.
        if case2_indices:
            keys2 = [batch.keys[i] for i in case2_indices]
            prompt_ids_list, real_lens, _ = build_reprefill_inputs(
                keys=keys2,
                partition_id=batch.partition_id,
                pad_id=self.tokenizer.pad_token_id,
            )
            results = self._reprefill_all(prompt_ids_list)
            nested = [
                slice_response_logprobs(
                    results[j].extra_fields["prompt_logprobs"],
                    real_lens[j][0],
                    real_lens[j][1],
                )
                for j in range(len(keys2))
            ]
            tq.kv_batch_put(
                keys=keys2,
                partition_id=batch.partition_id,
                fields=TensorDict({"new_rollout_log_probs": to_nested_jagged(nested)}, batch_size=len(keys2)),
            )
            for i in case2_indices:
                if batch.tags[i] is None:
                    batch.tags[i] = {}
                batch.tags[i]["resume_version"] = int(resume_version)

        # Case 3: copy rollout_log_probs → new_rollout_log_probs.
        if case3_indices:
            keys3 = [batch.keys[i] for i in case3_indices]
            rollout_lp = meta["rollout_log_probs"]
            nested3 = [rollout_lp[i] for i in case3_indices]
            tq.kv_batch_put(
                keys=keys3,
                partition_id=batch.partition_id,
                fields=TensorDict({"new_rollout_log_probs": to_nested_jagged(nested3)}, batch_size=len(keys3)),
            )
            for i in case3_indices:
                if batch.tags[i] is None:
                    batch.tags[i] = {}
                batch.tags[i]["resume_version"] = int(resume_version)

        # Case 1: nothing to do (client already wrote new_rollout_log_probs).

        metrics["partial_reprefill/case_distribution.case_1"] = float(case1_count)
        metrics["partial_reprefill/case_distribution.case_2"] = float(len(case2_indices))
        metrics["partial_reprefill/case_distribution.case_3"] = float(len(case3_indices))
        metrics["partial_reprefill/resume_version"] = float(resume_version)
        return batch

    def _compute_old_log_prob(self, batch, metrics: dict):
        # Identical to reprefill_decoupled: old_log_probs = new_rollout_log_probs.
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass:
            return super()._compute_old_log_prob(batch, metrics)
        compare = self.config.trainer.v1.partial_reprefill.get("compare_trainer_old_log_prob", False)
        if compare:
            batch = super()._compute_old_log_prob(batch, metrics)
            metrics["partial_reprefill/trainer_old_log_prob_computed"] = 1.0
        select_fields = ["new_rollout_log_probs"] + (["old_log_probs"] if compare else [])
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=select_fields,
        )
        if compare:
            data["trainer_old_log_probs"] = data.pop("old_log_probs")
        data["old_log_probs"] = data.pop("new_rollout_log_probs")
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
        metrics["partial_reprefill/old_log_prob_source"] = 1.0
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        from verl.trainer.ppo.v1.reprefill_utils import (
            compute_and_emit_staleness_metrics,
            compute_and_emit_token_staleness_metrics,
        )

        compute_and_emit_staleness_metrics(batch, metrics, self.global_steps)
        compute_and_emit_token_staleness_metrics(batch, metrics, self.global_steps)
        return super()._compute_advantage(batch, metrics)

    def _debug_log_prob_extra_fields(self) -> list[str]:
        return ["new_rollout_log_probs"]
