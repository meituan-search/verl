# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
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
"""Diagnostic: compare MAGI prefix-tree OLP against FA3 on the same batch."""

from __future__ import annotations

import logging

import torch

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

_lgg = logging.getLogger(__name__)


def cmp_magi_vs_fa3(log_probs: torch.Tensor, batch, config, compute_log_prob_fn) -> None:
    """Run a second OLP with FA3 on the same batch, then compare against MAGI.

    Logs mean abs diff, pos0 diff, and pos1+ diff.  Gated by
    ``actor_rollout_ref.model.use_prefix_tree=True``.
    """
    if not config.get("use_prefix_tree", False):
        return
    _lgg.warning("[MAGI-CMP] prefix_tree OLP done. Re-running with FA3 for comparison...")
    batch_td2 = batch.to_tensordict()
    batch_td2 = left_right_2_no_padding(batch_td2)
    tu.assign_non_tensor(
        batch_td2,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        compute_loss=False,
        use_prefix_tree=False,
    )
    try:
        out2 = compute_log_prob_fn(batch_td2)
        fa3_log_probs = no_padding_2_padding(tu.get(out2, "log_probs"), batch_td2)
    except RuntimeError:
        _lgg.warning("[MAGI-CMP] FA3 forward failed (expected if not supported), skipping compare")
        return

    mask = batch.batch["response_mask"].bool()
    magi_lp = log_probs[:, : fa3_log_probs.shape[-1]]
    fa3_lp = fa3_log_probs[:, : magi_lp.shape[-1]]
    diff = (magi_lp.float() - fa3_lp.float()).abs()
    p0 = diff[:, 0][mask[:, 0]].mean().item() if mask[:, 0].any() else 0
    pr = diff[:, 1:][mask[:, 1:]].mean().item() if mask[:, 1:].any() else 0
    all_ = diff[mask].mean().item() if mask.any() else 0
    _lgg.warning(
        "[MAGI-CMP] MAGI vs FA3 log_prob diff: mean=%.6f pos0=%.6f pos1+=%.6f",
        all_,
        p0,
        pr,
    )
