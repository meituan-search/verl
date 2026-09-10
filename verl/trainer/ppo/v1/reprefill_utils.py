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
"""Shared re-prefill helpers: compute response-token log probs of existing
trajectories under the rollout engine's current weight
(max_new_tokens=0, prompt_logprobs=0).

Used by staleness_sweep (diagnostics) and reprefill_decoupled (π_b for
Decoupled PPO)."""

import asyncio
import logging
import os

import numpy as np
import torch
import transfer_queue as tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def slice_response_logprobs(prompt_logprobs_ls, prompt_len, response_len):
    # SGLang prompt_logprobs_ls has length S = prompt_len + response_len
    # (only when logprob_start_len=0 — the case-2 full-reprefill path and
    # the distillation teacher). The piggyback resume path uses
    # logprob_start_len=prompt_len-1 and emits a separate
    # prefix_prompt_logprobs field instead; this helper does not apply
    # to that path (see _build_piggyback_fields in llm_server.py).
    # Entry i is the logprob of the token at position i+1 predicted by
    # tokens [0..i]. Response tokens occupy positions
    # [prompt_len, prompt_len + response_len - 1]; their logprobs are at
    # indices [prompt_len - 1, prompt_len + response_len - 2]. Each entry is
    # a single-element list when prompt_logprobs=0.
    start = max(prompt_len - 1, 0)
    end = prompt_len + response_len - 1
    return [float(entry[0]) for entry in prompt_logprobs_ls[start:end]]


def to_nested_jagged(nested_list):
    # Nested jagged layout is required for KVBatch.to_padded_tensor() to
    # recognize the field downstream (see trainer_staleness_sweep.py).
    return torch.nested.as_nested_tensor(
        [torch.tensor(lst, dtype=torch.float32) for lst in nested_list],
        layout=torch.jagged,
    )


def build_reprefill_inputs(keys, partition_id, pad_id):
    data = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["prompts", "responses"])
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


async def reprefill_trajectories(client, prompt_ids_list, request_prefix="reprefill"):
    sampling_params_list = [{"prompt_logprobs": 0, "max_new_tokens": 0}] * len(prompt_ids_list)
    results = await asyncio.gather(
        *[
            client.generate(
                request_id=f"{request_prefix}_{i}",
                prompt_ids=pids,
                sampling_params=sp,
            )
            for i, (pids, sp) in enumerate(zip(prompt_ids_list, sampling_params_list, strict=False))
        ]
    )
    return results


def compute_and_emit_staleness_metrics(batch, metrics, global_steps):
    """Emit offpolicy/* diagnostics from the three logprobs (rollout / new_rollout / old).

    Shared by staleness_sweep and reprefill_decoupled. Fetches fields from TQ;
    silently returns on fetch failure (metrics are best-effort).
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

    fields = ["rollout_log_probs", "new_rollout_log_probs", "old_log_probs", "response_mask"]
    try:
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
    except Exception as e:
        logger.warning(f"reprefill: failed to fetch logprobs for metrics: {e}")
        return

    from verl import DataProto

    data = DataProto(batch=data.to_padded_tensor())

    rollout_lp = data.batch["rollout_log_probs"]
    new_rollout_lp = data.batch["new_rollout_log_probs"]
    old_lp = data.batch["old_log_probs"]
    response_mask = data.batch["response_mask"]

    resume_versions = np.array(
        [tag.get("resume_version", global_steps - 1) for tag in batch.tags],
        dtype=np.int64,
    )
    staleness = (global_steps - 1) - resume_versions
    metrics["staleness_sweep/sample_staleness_mean"] = float(staleness.mean())
    metrics["staleness_sweep/sample_staleness_max"] = float(staleness.max())

    corr_metrics = compute_offpolicy_metrics(
        old_log_prob=old_lp,
        rollout_log_prob=rollout_lp,
        response_mask=response_mask,
        new_rollout_log_prob=new_rollout_lp,
    )
    for key, value in corr_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[f"offpolicy/{key}"] = value.item()
        else:
            metrics[f"offpolicy/{key}"] = value


def decide_case(
    piggyback_marker: bool,
    current_parameter_version: int,
    enable_case_skip: bool,
    enable_piggyback: bool,
    resume_version: int | None,
    max_resume_staleness: int = 0,
) -> int:
    """Decide which reprefill case to apply to a trajectory.

    Returns 1 (piggyback), 2 (full reprefill), or 3 (skip — copy rollout_log_probs).

    Keys on resume_version: the weight version the trajectory's existing
    logprobs were computed at (W_resume for piggyback, decode version for
    single-segment trajectories). Absent means the logprobs span multiple
    versions (resumed without piggyback) — always a full reprefill.
    """
    if resume_version is None:
        return 2
    if not enable_case_skip:
        return 2
    if current_parameter_version - resume_version > max_resume_staleness:
        return 2
    if piggyback_marker and enable_piggyback:
        return 1
    return 3


def build_token_versions(segment_versions: list[int], segment_lengths: list[int]) -> list[int]:
    """Expand per-segment version into a per-token list.

    Used by client to record each token's decode-time global_steps; carried as
    a list in extra_fields and converted to int32 tensor at the TQ promotion
    site, same as new_rollout_log_probs.
    """
    assert len(segment_versions) == len(segment_lengths)
    tokens: list[int] = []
    for v, n in zip(segment_versions, segment_lengths, strict=True):
        tokens.extend([int(v)] * n)
    return tokens


def build_partial_new_rollout_log_probs(
    prefix_prompt_logprobs: list[float],
    suffix_rollout_log_probs: list[float],
) -> list[float]:
    """Concatenate prefix (re-prefilled at W_resume) + suffix (decode logprob
    at W_resume, copied as-is) into one trajectory's new_rollout_log_probs.

    For partial_rollout trajectories: prefix tokens were decoded at W_prefix
    and re-prefilled at W_resume; suffix tokens were decoded at W_resume, so
    their rollout_log_probs IS the W_resume logprob — no re-prefill needed.
    """
    return list(prefix_prompt_logprobs) + list(suffix_rollout_log_probs)


def compute_and_emit_token_staleness_metrics(batch, metrics, global_steps):
    """Emit per-token staleness metrics using token_versions + the three logprobs.

    Metrics (all under `offpolicy_token/`):
      - staleness_mean: per-token mean of |log(π_rollout) - log(π_new_rollout)|
        over tokens where token_versions[i] < global_steps - 1.
      - fresh_token_ratio: fraction of tokens with version == global_steps - 1.
      - stale_token_ratio: complement.
      - staleness_by_version_gap_0: ratio of tokens with version gap 0.
      - staleness_by_version_gap_1: ratio with gap 1.
      - staleness_by_version_gap_2_3: ratio with gap 2-3.
      - staleness_by_version_gap_4plus: ratio with gap ≥4.

    Best-effort diagnostics: fetch failures and per-key missing/malformed
    `token_versions` (e.g. a case-1 piggyback trajectory in a mixed batch)
    skip that key's per-token stats without raising. Note that transfer_queue
    drops a field from the batch-level get result when ANY key lacks it, so a
    single key without `token_versions` suppresses these metrics for the
    whole batch.
    """
    fields = ["rollout_log_probs", "new_rollout_log_probs", "old_log_probs", "response_mask", "token_versions"]
    try:
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
    except Exception as e:
        logger.warning(f"token staleness metrics: failed to fetch: {e}")
        return

    token_versions_field = data.get("token_versions", None)

    current_version = global_steps - 1
    total_tokens = 0
    fresh_tokens = 0
    gap_buckets = {0: 0, 1: 0, "2_3": 0, "4plus": 0}
    staleness_diffs: list[float] = []

    for i in range(len(batch.keys)):
        try:
            rollout = data["rollout_log_probs"][i].tolist()
            new_rollout = data["new_rollout_log_probs"][i].tolist()
            mask = data["response_mask"][i].tolist()
            tv_entry = token_versions_field[i] if token_versions_field is not None else None
            tv = tv_entry.tolist() if tv_entry is not None else None
        except Exception as e:
            logger.debug(f"token staleness metrics: skipping key {batch.keys[i]} (malformed entry): {e}")
            continue
        if tv is None or len(tv) != len(rollout):
            # No usable token_versions for this key (mixed batch) or a
            # length mismatch (malformed) — skip this key's per-token stats.
            logger.debug(f"token staleness metrics: skipping key {batch.keys[i]} (no/mismatched token_versions)")
            continue
        for j in range(len(rollout)):
            if mask[j] == 0:
                continue
            total_tokens += 1
            gap = current_version - tv[j]
            if gap == 0:
                fresh_tokens += 1
                gap_buckets[0] += 1
            elif gap == 1:
                gap_buckets[1] += 1
            elif gap <= 3:
                gap_buckets["2_3"] += 1
            else:
                gap_buckets["4plus"] += 1
            if gap > 0:
                staleness_diffs.append(abs(rollout[j] - new_rollout[j]))

    if total_tokens == 0:
        return
    metrics["offpolicy_token/fresh_token_ratio"] = fresh_tokens / total_tokens
    metrics["offpolicy_token/stale_token_ratio"] = 1.0 - fresh_tokens / total_tokens
    metrics["offpolicy_token/staleness_mean"] = sum(staleness_diffs) / len(staleness_diffs) if staleness_diffs else 0.0
    metrics["offpolicy_token/staleness_by_version_gap_0"] = gap_buckets[0] / total_tokens
    metrics["offpolicy_token/staleness_by_version_gap_1"] = gap_buckets[1] / total_tokens
    metrics["offpolicy_token/staleness_by_version_gap_2_3"] = gap_buckets["2_3"] / total_tokens
    metrics["offpolicy_token/staleness_by_version_gap_4plus"] = gap_buckets["4plus"] / total_tokens
