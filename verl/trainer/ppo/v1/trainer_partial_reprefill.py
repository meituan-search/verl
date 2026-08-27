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

P2 (`enable_prefill_pipeline=true`): pre-dispatch re-prefills for newly
finished **case 2** trajectories while the replay buffer poll loop waits for
remaining generation; `on_sampled` then consumes version-aligned pending
entries and re-issues synchronously only for tail stragglers.

Enable via: trainer.v1.trainer_mode=partial_reprefill
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
from dataclasses import dataclass

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


@register_trainer("partial_reprefill")
class PPOTrainerPartialReprefill(PPOTrainerColocateAsync):
    """Partial reprefill trainer (colocate async)."""

    def on_train_begin(self):
        cfg = self.config.trainer.v1.partial_reprefill
        num_warmup_batches = cfg.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches (partial_reprefill)")
        if cfg.get("enable_prefill_pipeline", False):
            self._prefill_dispatcher = _PrefillDispatcher()
            self._prefill_dispatcher.start()
            self._pending_prefill: dict[str, _PendingPrefill] = {}
            self.replay_buffer.set_on_new_finished_callback(self._on_new_finished)
            logger.info("partial_reprefill: pipelined re-prefill pre-dispatch enabled")

    def on_train_end(self):
        # Shut the prefill dispatcher down if it was started (pipeline-
        # disabled path never sets `_prefill_dispatcher`).
        dispatcher = getattr(self, "_prefill_dispatcher", None)
        if dispatcher is not None:
            self._cancel_pending_prefills(reason="on_train_end")
            dispatcher.shutdown()
            logger.info("partial_reprefill: prefill dispatcher shut down on_train_end")

    def _on_new_finished(self, partition_id, new_keys):
        """Pre-dispatch re-prefills for newly finished trajectories — case-aware.

        Only case 2 (fully-stale) trajectories are pre-dispatched: case 1
        (piggyback) already carries new_rollout_log_probs from the client, and
        case 3 (fully-fresh) just copies rollout_log_probs — neither needs an
        engine round-trip. `new_keys` are prompt **uids**; resolve each to its
        trajectory keys via the freshly-synced replay buffer partition
        snapshot (the callback fires after `partitions` is populated — see
        trainer_reprefill_decoupled._on_new_finished).
        """
        if partition_id != "train":
            return
        cfg = self.config.trainer.v1.partial_reprefill
        current_version = self.global_steps - 1
        # Bound pre-dispatch to what the imminent sample() could select — at
        # most `train_batch_size` newly-finished uids per poll iteration,
        # oldest-first (smallest prompt_global_steps).
        train_batch_size = self.config.data.train_batch_size
        prompt_global_steps = self.replay_buffer.prompt_global_steps.get(partition_id, {})
        ordered_uids = sorted(new_keys, key=lambda u: prompt_global_steps.get(u, 0))
        bounded_uids = ordered_uids[:train_batch_size] if train_batch_size > 0 else ordered_uids
        if len(bounded_uids) < len(new_keys):
            logger.debug(
                f"partial_reprefill: pre-dispatch bounded to {len(bounded_uids)} "
                f"of {len(new_keys)} newly-finished uids (train_batch_size={train_batch_size})"
            )

        partition = self.replay_buffer.partitions.get(partition_id, {})
        for uid in bounded_uids:
            traj_keys = [k for k in partition if k.split("_")[0] == uid]
            for traj_key in traj_keys:
                if traj_key in self._pending_prefill:
                    continue
                # Case-awareness: fetch token_versions for this traj_key and
                # read the piggyback marker from the replay buffer partition
                # snapshot (synced from the TQ trajectory tag, which
                # agent_loop_tq populates for piggyback trajectories).
                try:
                    meta = tq.kv_batch_get(
                        keys=[traj_key],
                        partition_id=partition_id,
                        select_fields=["token_versions"],
                    )
                except Exception as e:
                    # Missing key or not-yet-ready fields: skip pre-dispatch —
                    # on_sampled will classify from the batch and re-issue
                    # synchronously if needed. Must not raise into the
                    # replay buffer poll callback that invokes us.
                    logger.debug(
                        f"partial_reprefill: case-detection fetch failed for {traj_key}: {e}; skipping pre-dispatch"
                    )
                    continue
                tag_dict = partition.get(traj_key) or {}
                piggyback = bool(tag_dict.get("piggyback_marker", False)) and cfg.enable_piggyback
                tv = meta.get("token_versions") if meta is not None else None
                last_tv = int(tv[0][-1].item()) if tv is not None and len(tv) > 0 and len(tv[0]) > 0 else None
                case = decide_case(
                    piggyback_marker=piggyback,
                    last_token_version=last_tv,
                    current_parameter_version=current_version,
                    enable_case_skip=cfg.enable_case_skip,
                    enable_piggyback=cfg.enable_piggyback,
                )
                if case != 2:
                    continue  # only pre-dispatch case 2
                try:
                    prompt_ids_list, _, _ = build_reprefill_inputs(
                        keys=[traj_key],
                        partition_id=partition_id,
                        pad_id=self.tokenizer.pad_token_id,
                    )
                    future = self._prefill_dispatcher.submit(
                        reprefill_trajectories(
                            client=self.get_llm_client(),
                            prompt_ids_list=prompt_ids_list,
                            request_prefix=f"partial_reprefill_p2_{self.global_steps}_{traj_key}",
                        )
                    )
                    self._pending_prefill[traj_key] = _PendingPrefill(
                        version=self.global_steps,
                        future=future,
                    )
                except Exception as e:
                    logger.warning(f"partial_reprefill: pre-dispatch failed for {traj_key}: {e}")

    def on_sampled(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        with marked_timer("new_rollout_log_prob", self.timing_raw, color="blue"):
            if getattr(self, "_pending_prefill", None) is not None:
                batch = self._compute_new_rollout_log_prob_pipelined(batch, metrics)
            else:
                batch = self._compute_new_rollout_log_prob(batch, metrics)
        return batch

    @auto_await
    async def _reprefill_all(self, prompt_ids_list):
        return await reprefill_trajectories(
            client=self.get_llm_client(),
            prompt_ids_list=prompt_ids_list,
            request_prefix=f"partial_reprefill_{self.global_steps}",
        )

    def _classify_cases(self, batch: KVBatchMeta, meta) -> tuple[list[int], list[int], int]:
        """Classify each trajectory in the batch into case 1/2/3.

        Returns (case2_indices, case3_indices, case1_count). `meta` is the
        kv_batch_get result holding at least `token_versions`.
        """
        cfg = self.config.trainer.v1.partial_reprefill
        current_version = self.global_steps - 1
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
        return case2_indices, case3_indices, case1_count

    def _reprefill_case2(
        self,
        keys: list[str],
        partition_id: str,
        pending: dict[str, _PendingPrefill] | None = None,
    ) -> tuple[list[list[float]], int]:
        """Run the full re-prefill for case 2 keys and slice response logprobs.

        When `pending` is given (P2 pipelined path), each key first tries its
        pre-dispatched future — consumed only if it exists, was issued at the
        current step, and resolved at the expected engine version; anything
        else falls back to a synchronous re-issue (batched over all fallback
        keys). Returns (nested response logprobs, number of consumed futures).
        """
        resume_version = self.global_steps - 1
        prompt_ids_list, real_lens, _ = build_reprefill_inputs(
            keys=keys,
            partition_id=partition_id,
            pad_id=self.tokenizer.pad_token_id,
        )

        results: list = [None] * len(keys)
        reissue_indices: list[int] = []
        consumed = 0
        for j, key in enumerate(keys):
            entry = pending.get(key) if pending is not None else None
            if entry is not None and entry.version == self.global_steps:
                try:
                    # One pre-dispatched request → 1-element result list.
                    result = entry.future.result(timeout=600)[0]
                    engine_version = result.extra_fields.get("global_steps", None)
                    if engine_version is not None and engine_version != resume_version:
                        logger.warning(
                            f"partial_reprefill: prefill version mismatch for {key} "
                            f"(engine={engine_version}, expected={resume_version}); re-issuing"
                        )
                    else:
                        results[j] = result
                        consumed += 1
                        continue
                except Exception as e:
                    logger.warning(f"partial_reprefill: prefill future failed for {key}: {e}")
            reissue_indices.append(j)

        if reissue_indices:
            keys_sub = [keys[j] for j in reissue_indices]
            prompt_ids_sub, _, _ = build_reprefill_inputs(
                keys=keys_sub,
                partition_id=partition_id,
                pad_id=self.tokenizer.pad_token_id,
            )
            reissue_results = self._reprefill_all(prompt_ids_sub)
            for k, j in enumerate(reissue_indices):
                results[j] = reissue_results[k]

        nested = [
            slice_response_logprobs(
                results[j].extra_fields["prompt_logprobs"],
                real_lens[j][0],
                real_lens[j][1],
            )
            for j in range(len(keys))
        ]
        return nested, consumed

    def _compute_new_rollout_log_prob(self, batch: KVBatchMeta, metrics: dict, _pending=None) -> KVBatchMeta:
        resume_version = self.global_steps - 1

        # Pre-fetch rollout_log_probs + token_versions for all keys (case 3 needs
        # rollout_log_probs; case dispatch needs token_versions[-1]).
        fields = ["rollout_log_probs", "token_versions"]
        meta = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        case2_indices, case3_indices, case1_count = self._classify_cases(batch, meta)

        # Case 2: full reprefill for selected keys (pipelined path consumes
        # pre-dispatched futures when `_pending` is provided).
        if case2_indices:
            keys2 = [batch.keys[i] for i in case2_indices]
            nested, consumed = self._reprefill_case2(
                keys2,
                batch.partition_id,
                pending=_pending,
            )
            if _pending is not None:
                metrics["partial_reprefill/prefill_consumed"] = float(consumed)
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

    def _cancel_pending_prefills(self, reason: str = "on_sampled") -> int:
        """Cancel unconsumed pre-dispatched re-prefill futures.

        `_pending_prefill.clear()` drops dict references, but the underlying
        coroutines on the dispatcher loop keep running; for unselected keys
        the request gets aborted by on_sample_end's abort_replicas, then the
        client's retry loop (should_retry=True for V1) retries it at the NEW
        weight and the result is silently discarded — holding engine
        capacity and LB sticky-session slots.

        `run_coroutine_threadsafe` returns a wrapper `Future` whose
        `cancel()` is safe to call: it stops the wrapper from running the
        coroutine's body if it hasn't started, and the coroutine (if already
        running) gets `CancelledError` at its next await.

        Returns the number of futures that were cancelled.
        """
        cancelled = 0
        for key, entry in list(self._pending_prefill.items()):
            if entry.future.done():
                continue
            try:
                if entry.future.cancel():
                    cancelled += 1
                else:
                    logger.debug(
                        f"partial_reprefill: prefill future for {key} "
                        f"could not be cancelled (already running/done) during {reason}"
                    )
            except Exception as e:
                logger.warning(f"partial_reprefill: error cancelling prefill future for {key}: {e}")
        if cancelled:
            logger.info(
                f"partial_reprefill: cancelled {cancelled} unconsumed pre-dispatched prefill futures during {reason}"
            )
        return cancelled

    def _compute_new_rollout_log_prob_pipelined(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """P2: case-aware consumption of pre-dispatched re-prefill futures.

        Keeps the case dispatcher from the non-pipelined path as the entry:
        case 2 trajectories prefer their pre-dispatched future (version-
        aligned, with a synchronous re-issue fallback for missing / stale /
        failed entries); case 1 and case 3 take the same fast paths as the
        non-pipelined path (no future lookup).
        """
        try:
            return self._compute_new_rollout_log_prob(batch, metrics, _pending=self._pending_prefill)
        finally:
            # Cancel unconsumed pre-dispatched futures (unselected keys,
            # case 1/3 keys, stale-version entries, mismatched-engine
            # entries) before clearing the dict so their coroutines don't
            # linger into the next step window where the client's
            # should_retry=True loop would re-dispatch them at the new
            # weight.
            cancelled = self._cancel_pending_prefills(reason="on_sampled")
            metrics["partial_reprefill/prefill_cancelled"] = float(cancelled)
            self._pending_prefill.clear()

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
