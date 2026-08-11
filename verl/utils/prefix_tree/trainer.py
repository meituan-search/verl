# Copyright 2025 Meituan Ltd. and/or its affiliates
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

Every public function here is a single call that checks *config* internally;
the caller never needs to gate on ``use_prefix_tree``.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from verl.utils.prefix_tree.dynamic import compute_prefix_tree_metrics, greedy_build_tries
from verl.utils.prefix_tree.segment_grouper import create_grpo_segment_metadata
from verl.utils.prefix_tree.tree import _is_prefix_tree_enabled, build_global_tree_from_segments


def apply_engine_config(engine_config, config_or_data: dict) -> None:
    """Thread prefix-tree flags from config into *engine_config*."""
    engine_config.use_prefix_tree = config_or_data.get("use_prefix_tree", False)
    engine_config.prefix_tree_attention = config_or_data.get("prefix_tree_attention", "magi")


def add_meta_info(meta_dict: dict, config_or_data: dict) -> None:
    """Add prefix-tree entries to a meta-info dict (mutates in-place)."""
    meta_dict["use_prefix_tree"] = config_or_data.get("use_prefix_tree", False)
    meta_dict["prefix_tree_attention"] = config_or_data.get("prefix_tree_attention", "magi")


def pt_metrics(
    metrics: dict,
    input_ids,  # TODO: use PrefixTrie / PrefixSubTrie
    config_or_data: dict,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    micro_batch_size: int = 0,
    trie=None,
    leaf_idx=None,
) -> None:
    """Compute prefix_tree/global_shared_ratio, packed_tokens, raw_tokens if use_prefix_tree enabled.

    Uses caller-provided trie (avoiding redundant greedy_build_tries). micro_batch_shared_ratio is NOT computed here
    (accurate version in prepare_prefix_tree_micro_batches, surfaced via maybe_collect_mbs_metric)."""
    if not _is_prefix_tree_enabled(config_or_data):
        return
    metrics.update(
        compute_prefix_tree_metrics(
            input_ids,
            attention_mask=attention_mask,
            max_token_len_per_gpu=max_token_len_per_gpu,
            micro_batch_size=micro_batch_size,
            trie=trie,
            leaf_idx=leaf_idx,
        )
    )


def attach_segment_metadata(batch, rollout_n: int) -> None:
    """Attach segment_hashes/segment_lengths for prefix-tree fast path (GRPO), from prompt UIDs and lengths.

    Skipped when branch-aware segment_hashes already set (e.g. tree_search_manager). Prompt-only fallback."""
    if rollout_n < 2:
        return
    prompt_uids = batch.non_tensor_batch.get("uid", None)
    if prompt_uids is None:
        return
    attention_mask = batch.batch["attention_mask"]
    response_length = batch.batch["responses"].size(1)
    prompt_lengths = attention_mask[:, :-response_length].sum(dim=-1).cpu().tolist()

    segment_hashes, segment_lengths = create_grpo_segment_metadata(
        prompt_uids=list(prompt_uids),
        prompt_lengths=prompt_lengths,
        rollout_n=rollout_n,
    )
    batch.non_tensor_batch["segment_hashes"] = segment_hashes
    batch.non_tensor_batch["segment_lengths"] = segment_lengths


def build_global_trie(batch, *, metrics=None, v1_tq=False) -> float:
    """Build global prefix trie, attach trie + leaf_idx to batch. Uses segment fast path if available, else greedy.

    Returns wall-clock seconds spent building trie (excluding input prep)."""
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch.get("attention_mask", None)
    if attention_mask is not None:
        seqs = [input_ids[i][attention_mask[i].bool()].tolist() or [0] for i in range(len(input_ids))]
    else:
        seqs = [input_ids[i].tolist() for i in range(len(input_ids))]
    total_raw = sum(len(s) for s in seqs)

    # if the tree metainfo is avaliable, don't need to build from scratch
    seg_hashes = batch.non_tensor_batch.get("segment_hashes", None)
    seg_lengths = batch.non_tensor_batch.get("segment_lengths", None)
    _t0 = time.perf_counter()
    if seg_hashes is not None and seg_lengths is not None:
        trie = build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)
    else:
        trie = None
    if trie is None:
        trie, _ = greedy_build_tries(seqs)
    _t1 = time.perf_counter()
    if metrics is not None:
        metrics["actor/prefix_tree/tree_build_time_s"] = _t1 - _t0
        batch.meta_info["prefix_tree_path_tag"] = "segment" if seg_hashes is not None else "uniform"
    if trie is None:
        return -1

    leaf_idx = np.full(len(seqs), -1, dtype=np.int64)
    # A sequence's leaf is the deepest node on its root->leaf path, i.e. the max
    # node_idx among nodes whose sequence_ids include it (along any path DFS
    # pre-order node_idx strictly increases with depth, so the max is the
    # sequence's end node). This correctly handles sequences that are strict
    # prefixes of others, which terminate at an internal (non-childless) node --
    # the previous childless-only scan orphaned those. Safe for the segment path
    # too (there every sequence ends at a childless leaf, which is also its max).
    for node_idx, node in enumerate(trie.nodes):
        for seq_id in node.sequence_ids:
            if node_idx > leaf_idx[seq_id]:
                leaf_idx[seq_id] = node_idx
    if (leaf_idx < 0).any():
        missing = np.where(leaf_idx < 0)[0].tolist()
        raise ValueError(
            f"build_global_trie: {len(missing)} samples have no leaf assigned "
            f"(first {missing[:8]}). The trie did not cover every sequence."
        )

    if v1_tq:
        import transfer_queue as tq

        from verl.utils import tensordict_utils as tu

        leaf_td = tu.get_tensordict({"leaf_idx": torch.from_numpy(leaf_idx)})
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=leaf_td)
        batch.extra_info["prefix_tree"] = trie
    else:
        batch.meta_info["prefix_tree"] = trie
        batch.batch["leaf_idx"] = torch.from_numpy(leaf_idx)
    return _t1 - _t0
