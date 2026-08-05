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
"""Staleness sweep trainer: one-shot rollout of N*train_batch, then N train steps.

Extends PPOTrainerSync with two additions:

1. `step()` cycle logic: at the start of each cycle (every N steps), submit
   N*train_batch prompts for one-shot rollout. Subsequent N-1 steps consume
   one train_batch chunk each from the pre-generated pool — no per-step
   rollout submission. After N steps, the pool is exhausted; the next step
   starts a new cycle (submit another N*train_batch).

2. `on_sampled()` full re-prefill: at the start of each step k (k=1..N within
   a cycle), the rollout engine has weight W_{k-1} (synced at the previous
   step's on_step_end). Reprefill ALL consumed trajectories' response tokens
   with W_{k-1} to obtain π_new-rollout, and write `new_rollout_log_probs` +
   `resume_version` tag to TransferQueue.

3. `_compute_advantage()` staleness/mismatch metrics: after `_compute_old_log_prob`
   populates `old_log_probs` (π_old at W_{k-1} on the train engine), fetch all
   three logprobs (rollout/new_rollout/old) and call `compute_offpolicy_metrics`
   to emit `staleness/*` + `mismatch/*` + `combined/*` metrics. PPO loss is
   unchanged (uses combined ratio; zero algorithm risk).

Each step k within a cycle has staleness = k-1 (samples were decoded at W_0,
current weights are W_{k-1}). Across N steps, sweep staleness 0..N-1.
"""
import asyncio
import logging
import os

import numpy as np
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.debug import marked_timer
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("staleness_sweep")
class PPOTrainerStalenessSweep(PPOTrainerSync):
    """Staleness sweep trainer.

    Cycles every N steps: one-shot rollout of N*train_batch at W_0, then N
    train steps consuming one train_batch each. At step k (k=1..N within a
    cycle), samples are (k-1)-step stale; reprefill at W_{k-1} measures
    staleness, old_log_prob at W_{k-1} measures combined off-policy gap,
    their diff measures pure T/R mismatch.

    Enable via:
        trainer.v1.trainer_mode=staleness_sweep
        trainer.v1.staleness_sweep.num_steps=N  (default 8)
    """

    def on_train_begin(self):
        # Initialize cycle counter; first step() will trigger rollout submission.
        self._steps_since_rollout = 0

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        # Cycle start: submit N*train_batch prompts for one-shot rollout.
        if self._steps_since_rollout == 0:
            num_steps = self.config.trainer.v1.staleness_sweep.num_steps
            for _ in range(num_steps):
                self._add_batch_to_generate()
            # Block until all N*train_batch trajectories are in TQ (decoded
            # at W_0). Subsequent steps then run normal sync PPO semantics:
            # sample chunk → on_sampled (reprefill at W_{k-1}) →
            # on_sample_end (sleep SGLang, safe because all generation is
            # done) → train → on_step_end (update weights to W_k).
            self._wait_for_cycle_rollout_complete()
            self._steps_since_rollout = num_steps
        self._steps_since_rollout -= 1

        # Single _step_once with full train_batch (parameter_sync_step must be 1).
        sample_batch_size = self.config.data.train_batch_size
        iter_metrics: dict = {}
        batch = self._step_once(iter_metrics, timing_raw, sample_batch_size)
        metrics.update(iter_metrics)
        return batch

    def on_sampled(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        # Full re-prefill at W_{global_steps-1} (rollout engine weight synced
        # at the previous step's on_step_end).
        with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
            batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    def _wait_for_cycle_rollout_complete(self) -> None:
        # Block until every prompt submitted at cycle start has finished (or
        # failed) generation. Without this, on_sample_end would call
        # sleep_replicas while SGLang still has N-1 chunks in-flight, hitting
        # sglang's "release_memory_occupation should be called only when no
        # ongoing request" assertion. After the wait, all N*train_batch
        # trajectories are in TQ at W_0, SGLang is idle, and the parent's
        # on_sample_end can safely sleep the engine each step.
        import time
        rb = self.replay_buffer
        last_debug = time.time()
        while True:
            rb._sync_metadata_from_transfer_queue()
            pending = len(rb.pending_keys.get("train", set()))
            running = len(rb.running_keys.get("train", set()))
            if pending == 0 and running == 0:
                break
            now = time.time()
            if now - last_debug > 30:
                logger.info(
                    f"staleness_sweep: waiting for cycle rollout to finish "
                    f"(pending={pending}, running={running})"
                )
                last_debug = now
            time.sleep(rb.poll_interval)
        finished = len(rb.finished_keys.get("train", set()))
        failure = len(rb.failure_keys.get("train", set()))
        logger.info(
            f"staleness_sweep: cycle rollout complete — "
            f"finished={finished}, failure={failure}, pending=0, running=0"
        )

    def _debug_log_prob_extra_fields(self) -> list[str]:
        # Expose new_rollout_log_probs (written in on_sampled) to
        # calculate_debug_metrics so Pair 2/3 (new_rollout vs rollout,
        # old vs new_rollout) are emitted under training/* prefix.
        return ["new_rollout_log_probs"]

    def _compute_new_rollout_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        # Reprefill all trajectories with current rollout engine weight.
        resume_version = self.global_steps - 1
        prompt_ids_list, real_lens, data = self._build_reprefill_inputs(
            keys=batch.keys, partition_id=batch.partition_id
        )
        sampling_params_list = [
            {"prompt_logprobs": 0, "max_new_tokens": 0}
        ] * len(batch.keys)
        results = self._reprefill_trajectories_async(prompt_ids_list, sampling_params_list)

        new_rollout_log_probs_nested: list[list[float]] = []
        for i, result in enumerate(results):
            prompt_len, response_len = real_lens[i]
            prompt_logprobs_ls = result.extra_fields["prompt_logprobs"]
            new_rollout_log_probs_nested.append(
                self._slice_response_logprobs(prompt_logprobs_ls, prompt_len, response_len)
            )

        # Store as nested jagged tensor (same format as rollout_log_probs /
        # old_log_probs in TQ) so KVBatch.to_padded_tensor() converts it
        # correctly downstream. NonTensorStack (from tu.get_tensordict) is
        # not recognized by to_padded_tensor, breaking torch.exp in
        # calculate_debug_metrics.
        data["new_rollout_log_probs"] = torch.nested.as_nested_tensor(
            [torch.tensor(lst, dtype=torch.float32) for lst in new_rollout_log_probs_nested],
            layout=torch.jagged,
        )
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=data.select("new_rollout_log_probs"),
        )
        for i in range(len(batch.keys)):
            if batch.tags[i] is None:
                batch.tags[i] = {}
            batch.tags[i]["resume_version"] = int(resume_version)

        metrics["staleness_sweep/resume_version"] = float(resume_version)
        metrics["staleness_sweep/steps_since_rollout"] = float(self._steps_since_rollout)
        return batch

    @auto_await
    async def _reprefill_trajectories_async(self, prompt_ids_list, sampling_params_list):
        client = self.get_llm_client()
        results = await asyncio.gather(*[
            client.generate(
                request_id=f"staleness_sweep_reprefill_{i}",
                prompt_ids=pids,
                sampling_params=sp,
            )
            for i, (pids, sp) in enumerate(zip(prompt_ids_list, sampling_params_list))
        ])
        return results

    def _build_reprefill_inputs(self, keys, partition_id):
        data = tq.kv_batch_get(
            keys=keys,
            partition_id=partition_id,
            select_fields=["prompts", "responses"],
        )
        pad_id = self.tokenizer.pad_token_id
        prompts_padded = data["prompts"].to_padded_tensor(padding=pad_id)
        responses_padded = data["responses"].to_padded_tensor(padding=pad_id)

        prompt_ids_list: list[list[int]] = []
        real_lens: list[tuple[int, int]] = []
        for i in range(len(keys)):
            prompt_ids = [int(x) for x in prompts_padded[i].tolist() if x != pad_id]
            response_ids = [int(x) for x in responses_padded[i].tolist() if x != pad_id]
            prompt_ids_list.append(prompt_ids + response_ids)
            real_lens.append((len(prompt_ids), len(response_ids)))
        return prompt_ids_list, real_lens, data

    def _slice_response_logprobs(self, prompt_logprobs_ls, prompt_len, response_len):
        # sglang prompt_logprobs_ls has length S = prompt_len + response_len.
        # Entry i is the logprob of the token at position i+1 predicted by
        # tokens [0..i]. Response tokens occupy positions
        # [prompt_len, prompt_len + response_len - 1]; their logprobs are at
        # indices [prompt_len - 1, prompt_len + response_len - 2] of
        # prompt_logprobs_ls. Each entry is a single-element list when
        # prompt_logprobs=0.
        start = max(prompt_len - 1, 0)
        end = prompt_len + response_len - 1
        return [float(entry[0]) for entry in prompt_logprobs_ls[start:end]]

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        # Emit staleness/mismatch/combined off-policy metrics from the three
        # logprobs, then delegate to parent for the actual advantage computation.
        # PPO loss is unchanged (parent uses combined ratio).
        self._compute_staleness_metrics(batch, metrics)
        return super()._compute_advantage(batch, metrics)

    def _compute_staleness_metrics(self, batch: KVBatchMeta, metrics: dict) -> None:
        # Fetch all three logprobs + response_mask from TQ. new_rollout_log_probs
        # was written in on_sampled; old_log_probs in _compute_old_log_prob.
        fields = ["rollout_log_probs", "new_rollout_log_probs", "old_log_probs", "response_mask"]
        try:
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=fields,
            )
        except Exception as e:
            logger.warning(f"staleness_sweep: failed to fetch logprobs for metrics: {e}")
            return

        from verl import DataProto
        data = DataProto(batch=data.to_padded_tensor())

        rollout_lp = data.batch["rollout_log_probs"]
        new_rollout_lp = data.batch["new_rollout_log_probs"]
        old_lp = data.batch["old_log_probs"]
        response_mask = data.batch["response_mask"]

        # resume_version (set in on_sampled) vs implicit consume_version
        # (= global_steps, since _compute_old_log_prob runs at this step).
        resume_versions = np.array(
            [tag.get("resume_version", self.global_steps - 1) for tag in batch.tags],
            dtype=np.int64,
        )
        staleness = (self.global_steps - 1) - resume_versions
        metrics["staleness_sweep/sample_staleness_mean"] = float(staleness.mean())
        metrics["staleness_sweep/sample_staleness_max"] = float(staleness.max())

        corr_metrics = compute_offpolicy_metrics(
            old_log_prob=old_lp,
            rollout_log_prob=rollout_lp,
            response_mask=response_mask,
            new_rollout_log_prob=new_rollout_lp,
        )
        # Wrap with "offpolicy/" prefix to group all compute_offpolicy_metrics
        # outputs (combined group + staleness/* + mismatch/*) under one
        # namespace. staleness_sweep/* meta metrics stay unwrapped.
        for key, value in corr_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[f"offpolicy/{key}"] = value.item()
            else:
                metrics[f"offpolicy/{key}"] = value
