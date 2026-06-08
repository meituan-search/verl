# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Prefix-tree helpers consumed by verl trainers (SFT, PPO).

Every public function here is a single call that checks *config* internally —
the caller never needs to gate on ``use_prefix_tree``.
"""

from __future__ import annotations

import numpy as np

# ──────────────────────── engine config ──────────────────────────


def apply_engine_config(engine_config, config_or_data: dict) -> None:
    """Thread prefix-tree flags from config into *engine_config*."""
    engine_config.use_prefix_tree = config_or_data.get("use_prefix_tree", False)
    engine_config.prefix_tree_attention = config_or_data.get("prefix_tree_attention", "flex")


# ──────────────────────── meta info ─────────────────────────────


def add_meta_info(meta_dict: dict, config_or_data: dict) -> None:
    """Add prefix-tree entries to a meta-info dict (mutates in-place)."""
    meta_dict["use_prefix_tree"] = config_or_data.get("use_prefix_tree", False)
    meta_dict["prefix_tree_attention"] = config_or_data.get("prefix_tree_attention", "flex")


# ──────────────────────── metrics ───────────────────────────────


def compute_metrics(metrics: dict, input_ids, config_or_data: dict) -> None:
    """Compute prefix-sharing metrics if *use_prefix_tree* is enabled.

    Updates *metrics* in-place with keys like ``prefix_tree/sharing_ratio``.
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return
    from verl.utils.prefix_tree.dynamic import compute_prefix_tree_metrics

    metrics.update(compute_prefix_tree_metrics(input_ids))


# ──────────────────────── prefix segments injection ──────────────


def inject_prefix_segments(gen_batch_output, config_or_data: dict) -> None:
    """Build ``prefix_segments`` for prefix-tree MAGI attention after generation.

    Mutates ``gen_batch_output.non_tensor_batch["prefix_segments"]``.
    No-op when *use_prefix_tree* is disabled or segments already exist.
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return
    if gen_batch_output.batch is None or "input_ids" not in gen_batch_output.batch.keys():
        return

    if "prefix_segments" in gen_batch_output.non_tensor_batch:
        return

    from verl.utils.prefix_tree.magi import build_prefix_segments_single_turn

    _ids = gen_batch_output.batch["input_ids"]
    _mask = gen_batch_output.batch.get("attention_mask", None)
    rollout_n = config_or_data.get("n", 1)

    # Compute once per unique prompt (before repeat), then repeat.
    if hasattr(gen_batch_output, "_orig_ids"):
        _orig_ids = gen_batch_output._orig_ids
        _orig_mask = gen_batch_output._orig_mask
    else:
        _orig_ids = gen_batch_output.batch["input_ids"]
        _orig_mask = gen_batch_output.batch.get("attention_mask", None)

    unique_segs = np.array(
        [
            build_prefix_segments_single_turn(_orig_ids[i], _orig_mask[i] if _orig_mask is not None else None)
            for i in range(_orig_ids.shape[0])
        ],
        dtype=object,
    )
    gen_batch_output.non_tensor_batch["prefix_segments"] = np.repeat(unique_segs, repeats=rollout_n, axis=0)


# ──────────────────────── disable for log-prob ───────────────────


def disable_for_log_prob(batch_td, config_or_data: dict, micro_batch_size_per_gpu: int) -> None:
    """Disable prefix-tree and dynamic bsz for log-prob computation pass."""
    if not _is_prefix_tree_enabled(config_or_data):
        return
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(
        batch_td,
        use_prefix_tree=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
    )


# ──────────────────────── internal ──────────────────────────────


def _is_prefix_tree_enabled(config_or_data) -> bool:
    if isinstance(config_or_data, dict):
        return config_or_data.get("use_prefix_tree", False)
    return bool(getattr(config_or_data, "use_prefix_tree", False))
