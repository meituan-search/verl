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
    # sglang prompt_logprobs_ls has length S = prompt_len + response_len.
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
    data = tq.kv_batch_get(
        keys=keys, partition_id=partition_id, select_fields=["prompts", "responses"]
    )
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
    results = await asyncio.gather(*[
        client.generate(
            request_id=f"{request_prefix}_{i}",
            prompt_ids=pids,
            sampling_params=sp,
        )
        for i, (pids, sp) in enumerate(zip(prompt_ids_list, sampling_params_list))
    ])
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
            keys=batch.keys, partition_id=batch.partition_id, select_fields=fields,
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
