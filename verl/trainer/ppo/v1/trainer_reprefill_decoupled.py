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
"""Re-prefill Decoupled PPO trainer: π_b (old_log_probs) from rollout-side re-prefill.

Extends PPOTrainerColocateAsync:
1. `on_sampled()`: re-prefill all consumed trajectories on the rollout engine
   (which holds W_{k-1} throughout the step window) and write
   `new_rollout_log_probs` + `resume_version` tags to TransferQueue.
2. `_compute_old_log_prob()`: copy `new_rollout_log_probs` as `old_log_probs`
   — no trainer-side forward pass. Downstream TIS/MIS/RS correction
   (compute_rollout_correction_and_add_to_batch) and the actor loss are
   unchanged; they consume `old_log_probs` as π_b regardless of origin.

P2 (`enable_prefill_pipeline=true`): pre-dispatch re-prefills for newly
finished trajectories while the replay buffer poll loop waits for remaining
generation. `on_sampled` then consumes valid pending entries (version-
aligned with the current step window) and only re-issues synchronously for
tail stragglers.

Enable via: trainer.v1.trainer_mode=reprefill_decoupled
"""
import asyncio
import concurrent.futures
import logging
import os
import threading
from dataclasses import dataclass

import transfer_queue as tq

from verl.trainer.ppo.v1.reprefill_utils import (
    build_reprefill_inputs,
    compute_and_emit_staleness_metrics,
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


@dataclass
class _PendingPrefill:
    version: int  # trainer.global_steps when the re-prefill was issued
    future: concurrent.futures.Future  # result: 1-element list of engine results


class _PrefillDispatcher:
    """Background event-loop thread for issuing re-prefill requests off the
    trainer thread (called from the replay buffer poll loop)."""

    def __init__(self):
        self._loop = None
        self._thread = None

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
            self._loop = None
            self._thread = None


@register_trainer("reprefill_decoupled")
class PPOTrainerReprefillDecoupled(PPOTrainerColocateAsync):
    """Re-prefill Decoupled PPO trainer (colocate async)."""

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.reprefill_decoupled.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")
        if self.config.trainer.v1.reprefill_decoupled.get("enable_prefill_pipeline", False):
            self._prefill_dispatcher = _PrefillDispatcher()
            self._prefill_dispatcher.start()
            self._pending_prefill: dict[str, _PendingPrefill] = {}
            self.replay_buffer.set_on_new_finished_callback(self._on_new_finished)
            logger.info("reprefill_decoupled: pipelined re-prefill pre-dispatch enabled")

    def _on_new_finished(self, partition_id, new_keys):
        # Pre-dispatch a single-request re-prefill for each newly finished
        # uid. `new_keys` is a set of prompt uids (the prompt key, not the
        # trajectory key) — matches `_pending_prefill`'s uid keying.
        if partition_id != "train":
            return
        for key in new_keys:
            if key in self._pending_prefill:
                continue
            try:
                prompt_ids_list, _, _ = build_reprefill_inputs(
                    keys=[key], partition_id=partition_id,
                    pad_id=self.tokenizer.pad_token_id,
                )
                future = self._prefill_dispatcher.submit(
                    reprefill_trajectories(
                        client=self.get_llm_client(),
                        prompt_ids_list=prompt_ids_list,
                        request_prefix=f"reprefill_p2_{self.global_steps}_{key}",
                    )
                )
                self._pending_prefill[key] = _PendingPrefill(
                    version=self.global_steps, future=future
                )
            except Exception as e:
                logger.warning(f"reprefill_decoupled: pre-dispatch failed for {key}: {e}")

    def on_sampled(self, batch: "KVBatchMeta", metrics: dict) -> "KVBatchMeta":
        if getattr(self, "_pending_prefill", None) is not None:
            with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
                batch = self._compute_new_rollout_log_prob_pipelined(batch, metrics)
        else:
            with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
                batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    @auto_await
    async def _reprefill_all(self, prompt_ids_list):
        return await reprefill_trajectories(
            client=self.get_llm_client(),
            prompt_ids_list=prompt_ids_list,
            request_prefix=f"reprefill_decoupled_{self.global_steps}",
        )

    def _compute_new_rollout_log_prob(self, batch, metrics):
        resume_version = self.global_steps - 1
        prompt_ids_list, real_lens, data = build_reprefill_inputs(
            keys=batch.keys, partition_id=batch.partition_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        results = self._reprefill_all(prompt_ids_list)
        nested = [
            slice_response_logprobs(
                results[i].extra_fields["prompt_logprobs"], real_lens[i][0], real_lens[i][1]
            )
            for i in range(len(batch.keys))
        ]
        data["new_rollout_log_probs"] = to_nested_jagged(nested)
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id,
            fields=data.select("new_rollout_log_probs"),
        )
        for i in range(len(batch.keys)):
            if batch.tags[i] is None:
                batch.tags[i] = {}
            batch.tags[i]["resume_version"] = int(resume_version)
        metrics["reprefill_decoupled/resume_version"] = float(resume_version)
        return batch

    def _compute_new_rollout_log_prob_pipelined(self, batch, metrics):
        # `_pending_prefill` is keyed by prompt uid (what the callback
        # delivers), but `batch.keys` are trajectory keys
        # (`{uid}_{session}_{index}`). Derive the uid via `key.split("_")[0]`
        # — the established convention in replay_buffer.py. A single
        # pre-dispatched request's 1-element result only covers one
        # trajectory; if a uid maps to multiple trajectory keys in this
        # batch (multi-session or multi-turn), the pre-dispatched result
        # can't be reused for both — re-issue those synchronously.
        resume_version = self.global_steps - 1
        prompt_ids_list, real_lens, data = build_reprefill_inputs(
            keys=batch.keys, partition_id=batch.partition_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        uid_to_indices: dict[str, list[int]] = {}
        for i, key in enumerate(batch.keys):
            uid_to_indices.setdefault(key.split("_")[0], []).append(i)

        results_by_index: dict[int, object] = {}
        reissue_indices: list[int] = []
        for i, key in enumerate(batch.keys):
            uid = key.split("_")[0]
            entry = self._pending_prefill.get(uid)
            consumed = False
            if (
                entry is not None
                and entry.version == self.global_steps
                and len(uid_to_indices[uid]) == 1
            ):
                try:
                    result = entry.future.result(timeout=600)[0]
                    engine_version = result.extra_fields.get("global_steps", None)
                    if engine_version is not None and engine_version != resume_version:
                        logger.warning(
                            f"reprefill_decoupled: prefill version mismatch for {key} "
                            f"(engine={engine_version}, expected={resume_version}); re-issuing"
                        )
                    else:
                        results_by_index[i] = result
                        consumed = True
                except Exception as e:
                    logger.warning(
                        f"reprefill_decoupled: prefill future failed for {key}: {e}"
                    )
            if not consumed:
                reissue_indices.append(i)

        if reissue_indices:
            keys = [batch.keys[i] for i in reissue_indices]
            prompt_ids_sub, _, _ = build_reprefill_inputs(
                keys=keys, partition_id=batch.partition_id,
                pad_id=self.tokenizer.pad_token_id,
            )
            reissue_results = self._reprefill_all(prompt_ids_sub)
            for j, i in enumerate(reissue_indices):
                results_by_index[i] = reissue_results[j]

        nested = [
            slice_response_logprobs(
                results_by_index[i].extra_fields["prompt_logprobs"],
                real_lens[i][0], real_lens[i][1],
            )
            for i in range(len(batch.keys))
        ]
        data["new_rollout_log_probs"] = to_nested_jagged(nested)
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id,
            fields=data.select("new_rollout_log_probs"),
        )
        for i in range(len(batch.keys)):
            if batch.tags[i] is None:
                batch.tags[i] = {}
            batch.tags[i]["resume_version"] = int(resume_version)
        metrics["reprefill_decoupled/resume_version"] = float(resume_version)
        # Entries for unselected keys would be wrong-version next step anyway;
        # drop all.
        self._pending_prefill.clear()
        return batch

    def _compute_old_log_prob(self, batch, metrics: dict):
        # Bypass mode stays available for A/B experiments: parent copies
        # rollout_log_probs as old_log_probs (2-policy semantics).
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass:
            return super()._compute_old_log_prob(batch, metrics)

        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=["new_rollout_log_probs"],
        )
        data["old_log_probs"] = data.pop("new_rollout_log_probs")
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data
        )
        metrics["reprefill_decoupled/old_log_prob_source"] = 1.0  # 1.0 = reprefill
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        # offpolicy/* diagnostics (staleness / mismatch / combined) from the
        # three logprobs; then parent computes the actual advantage. PPO loss
        # is unchanged (parent uses the combined ratio).
        compute_and_emit_staleness_metrics(batch, metrics, self.global_steps)
        return super()._compute_advantage(batch, metrics)

    def _debug_log_prob_extra_fields(self) -> list[str]:
        # Expose new_rollout_log_probs to calculate_debug_metrics.
        return ["new_rollout_log_probs"]
