# Copyright 2025 Individual Contributor: TomQunChaoA
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

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate pairwise logprobs diff between rollout / new_rollout / old (actor),
    for debugging purpose. Three pairwise comparisons when new_rollout_log_probs
    is available (staleness_sweep / 4policy Phase 1):

    - rollout vs old           → training/rollout_probs_diff_*           (combined)
    - new_rollout vs rollout   → training/new_rollout_vs_rollout_probs_diff_*  (staleness)
    - old vs new_rollout       → training/old_vs_new_rollout_probs_diff_*      (mismatch)

    When new_rollout_log_probs is absent, only the combined comparison is emitted
    (backward compatible with the original behavior).

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            new_rollout_log_probs (optional): log_probs from re-prefill at current
                rollout engine weight (π_new-rollout); enables staleness/mismatch
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor
            (when new_rollout_log_probs present, also:)
            "training/new_rollout_vs_rollout_probs_diff_{valid,max,mean,std}"
            "training/new_rollout_rollout_probs_pearson_corr"
            "training/old_vs_new_rollout_probs_diff_{valid,max,mean,std}"
            "training/old_new_rollout_probs_pearson_corr"
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    has_new_rollout = "new_rollout_log_probs" in data.batch
    new_rollout_log_probs = data.batch.get("new_rollout_log_probs", None) if has_new_rollout else None
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    new_rollout_probs = torch.exp(new_rollout_log_probs) if new_rollout_log_probs is not None else None
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        metrics = {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }
        if new_rollout_probs is not None:
            metrics.update({
                "training/new_rollout_vs_rollout_probs_diff_valid": 0,
                "training/new_rollout_vs_rollout_probs_diff_max": float("nan"),
                "training/new_rollout_vs_rollout_probs_diff_mean": float("nan"),
                "training/new_rollout_vs_rollout_probs_diff_std": float("nan"),
                "training/new_rollout_rollout_probs_pearson_corr": float("nan"),
                "training/old_vs_new_rollout_probs_diff_valid": 0,
                "training/old_vs_new_rollout_probs_diff_max": float("nan"),
                "training/old_vs_new_rollout_probs_diff_mean": float("nan"),
                "training/old_vs_new_rollout_probs_diff_std": float("nan"),
                "training/old_new_rollout_probs_pearson_corr": float("nan"),
            })
        return metrics

    # Pair 1: rollout vs old (actor) — combined off-policy gap (always emitted)
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }

    # Pair 2 & 3: only when π_new-rollout (new_rollout_log_probs) is available
    if new_rollout_probs is not None:
        # Pair 2: new_rollout vs rollout — staleness (weight drift since decode time)
        new_rollout_rollout_diff = calculate_log_prob_diff(new_rollout_probs, rollout_probs, response_mask_bool)
        new_rollout_rollout_pearson = pearson_correlation_coefficient(
            new_rollout_probs, rollout_probs, response_mask_bool
        )
        metrics["training/new_rollout_vs_rollout_probs_diff_valid"] = 1
        metrics["training/new_rollout_vs_rollout_probs_diff_max"] = torch.max(new_rollout_rollout_diff).detach().item()
        metrics["training/new_rollout_vs_rollout_probs_diff_mean"] = torch.mean(new_rollout_rollout_diff).detach().item()
        metrics["training/new_rollout_vs_rollout_probs_diff_std"] = torch.std(new_rollout_rollout_diff).detach().item()
        metrics["training/new_rollout_rollout_probs_pearson_corr"] = new_rollout_rollout_pearson

        # Pair 3: old vs new_rollout — T/R engine mismatch at same weight
        old_new_rollout_diff = calculate_log_prob_diff(actor_probs, new_rollout_probs, response_mask_bool)
        old_new_rollout_pearson = pearson_correlation_coefficient(
            actor_probs, new_rollout_probs, response_mask_bool
        )
        metrics["training/old_vs_new_rollout_probs_diff_valid"] = 1
        metrics["training/old_vs_new_rollout_probs_diff_max"] = torch.max(old_new_rollout_diff).detach().item()
        metrics["training/old_vs_new_rollout_probs_diff_mean"] = torch.mean(old_new_rollout_diff).detach().item()
        metrics["training/old_vs_new_rollout_probs_diff_std"] = torch.std(old_new_rollout_diff).detach().item()
        metrics["training/old_new_rollout_probs_pearson_corr"] = old_new_rollout_pearson

    return metrics
