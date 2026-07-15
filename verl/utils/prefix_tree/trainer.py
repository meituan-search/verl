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

"""Prefix-tree helpers consumed by verl trainers (SFT, PPO).

Every public function here is a single call that checks *config* internally —
the caller never needs to gate on ``use_prefix_tree``.
"""

from __future__ import annotations

from verl.utils.prefix_tree.dynamic import compute_prefix_tree_metrics
from verl.utils.prefix_tree.tree import _is_prefix_tree_enabled


def apply_engine_config(engine_config, config_or_data: dict) -> None:
    """Thread prefix-tree flags from config into *engine_config*."""
    engine_config.use_prefix_tree = config_or_data.get("use_prefix_tree", False)
    engine_config.prefix_tree_attention = config_or_data.get("prefix_tree_attention", "flex")


def add_meta_info(meta_dict: dict, config_or_data: dict) -> None:
    """Add prefix-tree entries to a meta-info dict (mutates in-place)."""
    meta_dict["use_prefix_tree"] = config_or_data.get("use_prefix_tree", False)
    meta_dict["prefix_tree_attention"] = config_or_data.get("prefix_tree_attention", "flex")


def pt_metrics(
    metrics: dict,
    input_ids,  # TODO: use PrefixTrie / PrefixSubTrie
    config_or_data: dict,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    micro_batch_size: int = 0,
) -> None:
    """Compute prefix-sharing metrics if *use_prefix_tree* is enabled.

    Updates *metrics* in-place with keys like ``prefix_tree/sharing_ratio``.
    Pass *attention_mask* to strip padding from 2-D padded tensors.
    Pass *max_token_len_per_gpu* for dynbsz (trie-based micro-batch groups).
    Pass *micro_batch_size* for fixed-mbs (consecutive-slice groups).
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return
    metrics.update(
        compute_prefix_tree_metrics(  # TODO: use PrefixTrie / PrefixSubTrie
            input_ids,
            attention_mask=attention_mask,
            max_token_len_per_gpu=max_token_len_per_gpu,
            micro_batch_size=micro_batch_size,
        )
    )
