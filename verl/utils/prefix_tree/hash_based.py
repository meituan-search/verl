# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Hash-based prefix detection.

Counterpart to :func:`verl.utils.prefix_tree.dynamic.build_tree_dynamic` —
both produce the same ``(TreeNode, leaf_to_sample)`` contract consumed by
:func:`verl.utils.prefix_tree.utils.build_layout_from_tree_node`.

Two-stage detection:
  1. Root prefix length — from per-turn hashes (``prefix_segments_batch``)
     when available, else from a token-level LCP scan.
  2. Multi-level (depth-2) — when ``prefix_segments_batch`` is provided and
     the batch has ≥2 groups of ≥2 samples sharing a second-turn hash,
     produce a depth-2 tree; otherwise fall back to single-level (root +
     per-sample leaves).

Supports depth-1 and depth-2 only. For arbitrary-depth trees, use
``build_tree_dynamic``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from torch import Tensor

from verl.utils.prefix_tree.utils import TreeNode, longest_common_prefix_length

__all__ = [
    "_hash_prefix",
    "build_prefix_segments_single_turn",
    "build_tree_hash_based",
]


def _hash_prefix(token_ids_flat: Tensor) -> int:
    """128-bit hash of a 1-D token-id tensor (full cumulative prefix).

    Uses xxhash when available (faster); falls back to hashlib.md5.
    The 128-bit width makes accidental collision negligible in practice.
    """
    raw = token_ids_flat.numpy().tobytes()
    try:
        import xxhash  # type: ignore[import]

        return xxhash.xxh128_intdigest(raw)
    except ImportError:
        import hashlib

        return int.from_bytes(hashlib.md5(raw).digest(), "little")


def build_prefix_segments_single_turn(
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> list[tuple[int, int]]:
    """Build a single-entry prefix_segments list for one sample.

    Used by RL trainers when per-sub-turn boundaries are unavailable — the
    single entry covers the entire real (non-pad) prompt and lets the hash
    path detect the prompt as a shared root prefix across GRPO-style
    rollouts.

    Args:
        input_ids: 1-D or 2-D (1, seq_len) token tensor.
        attention_mask: Optional 1-D or 2-D mask; when provided, only the
            tokens where mask==1 are considered (strips padding).

    Returns:
        ``[(hash, prompt_len)]`` — a one-element prefix_segments list.
    """
    ids = input_ids.flatten()
    if attention_mask is not None:
        mask = attention_mask.flatten().bool()
        ids = ids[mask]
    h = _hash_prefix(ids.cpu())
    return [(h, int(ids.numel()))]


def build_tree_hash_based(
    samples: list[Tensor],
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
) -> Optional[tuple[TreeNode, list[int]]]:
    """Hash-based prefix detection. Returns ``(TreeNode, leaf_to_sample)`` or None.

    ``leaf_to_sample[i]`` gives the original sample index for the i-th leaf in
    DFS pre-order. Returns ``None`` when no shared prefix exists.
    """
    import os as _os

    n = len(samples)
    if n == 0:
        return None

    if prefix_segments_batch is not None and len(prefix_segments_batch) == n:
        prefix_len = _resolve_prefix_len_from_segments(prefix_segments_batch)
        if _os.environ.get("DEBUG_PREFIX_LEN") == "1":
            scan_len = longest_common_prefix_length(samples)
            T = samples[0].shape[0] if samples else 0
            print(
                f"[PREFIX_LEN] seg_len={prefix_len} scan_len={scan_len} T={T} "
                f"n_segs={[len(s) for s in prefix_segments_batch]}",
                flush=True,
            )
        # If segments share no global prefix, fall through to LCP so that
        # _resolve_multilevel_tree can still detect per-group sub-prefixes
        # (e.g. GRPO: n=4 rollouts per prompt, multiple prompt groups per mbs).
        if prefix_len == 0:
            prefix_len = longest_common_prefix_length(samples)
    else:
        prefix_len = longest_common_prefix_length(samples)

    if prefix_len == 0:
        return None

    if prefix_segments_batch is not None:
        actual_root_len = longest_common_prefix_length(samples)
        if actual_root_len > 0:
            multilevel = _resolve_multilevel_tree(samples, prefix_segments_batch, actual_root_len)
            if multilevel is not None:
                root_len, children_info = multilevel  # [(idxs, group_TreeNode), ...]
                leaf_to_sample = [int(idx) for idxs, _ in children_info for idx in idxs]
                # Only use multilevel if ALL n sequences are covered; otherwise a
                # partial group (single-member prompt) would cause restore_flat_to_nested
                # to leave some sample_tensors[i] as None → AssertionError.
                if len(set(leaf_to_sample)) == n:
                    root = TreeNode(segment_len=root_len, children=[g for _, g in children_info])
                    return root, leaf_to_sample
                # else: fall through to single-level fallback

    # Single-level fallback: root + per-sample leaves (one leaf per sample).
    leaves = [TreeNode(segment_len=int(t.shape[0]) - prefix_len) for t in samples]
    root = TreeNode(segment_len=prefix_len, children=leaves)
    return root, list(range(n))


def _resolve_prefix_len_from_segments(
    prefix_segments_batch: list[list[tuple[int, int]]],
) -> int:
    """Return the longest shared-prefix length derivable from per-sample segment lists.

    Each element of ``prefix_segments_batch`` is a list of
    ``(hash, cumulative_len)`` pairs produced by the dataset at load time. Two
    samples share turn k when all samples have the same per-turn hash at
    position k.

    Walks turn-by-turn and stops at the first turn where hashes diverge;
    returns the cumulative_len from sample 0 at the last shared turn (0 if
    none).

    Compares only hashes (not cum_len) because tokenization boundary effects
    can shift cum_len slightly between samples even for identical turns.
    """
    n = len(prefix_segments_batch)
    if n == 0:
        return 0

    min_turns = min(len(segs) for segs in prefix_segments_batch)
    if min_turns == 0:
        return 0

    best = 0
    for turn_idx in range(min_turns):
        h0 = prefix_segments_batch[0][turn_idx][0]
        if all(prefix_segments_batch[i][turn_idx][0] == h0 for i in range(1, n)):
            best = prefix_segments_batch[0][turn_idx][1]
        else:
            break
    return best


def _resolve_multilevel_tree(
    tokens_by_sample: list,
    prefix_segments_batch: list[list[tuple[int, int]]],
    root_prefix_len: int,
) -> Optional[tuple[int, list[tuple[list[int], TreeNode]]]]:
    """Detect a depth-2 tree from prefix_segments. Returns ``(root_len, children_info)`` or None.

    Groups samples by their first post-root segment hash. If ≥2 groups each
    have ≥2 samples, a depth-2 tree exists and we return
    ``(root_len, [(sample_idxs, group_TreeNode), ...])``.
    """
    n = len(tokens_by_sample)
    if prefix_segments_batch is None or n < 4:
        return None

    # Group samples by the hash of their first post-root turn (O(batch×turns)).
    groups: dict[int, list[int]] = defaultdict(list)
    for i, segs in enumerate(prefix_segments_batch):
        next_seg = next((s for s in segs if s[1] > root_prefix_len), None)
        if next_seg is None:
            return None
        groups[next_seg[0]].append(i)

    # Need ≥2 groups each with ≥2 samples for multi-level to be worthwhile.
    useful = [(h, idxs) for h, idxs in groups.items() if len(idxs) >= 2]
    if len(useful) < 2:
        return None

    # Sort children by first sequence index so DFS traversal order matches
    # the flat-trie token layout (build_layout_from_tree_node expects
    # non-decreasing leaf ranges — violating this raises ValueError).
    useful.sort(key=lambda x: min(x[1]))

    # Use token scan (not segment hashes) to get exact turn2 shared prefix length
    # per group; segment hashes can mis-align due to chat-template boundary effects.
    children = []
    for _h, idxs in useful:
        group_tokens = [tokens_by_sample[i] for i in idxs]
        suffixes = [t[root_prefix_len:] for t in group_tokens]
        shared_suffix_len = longest_common_prefix_length(suffixes)
        if shared_suffix_len <= 0:
            return None
        group_turn2_len = shared_suffix_len
        leaves = [TreeNode(int(tokens_by_sample[i].shape[0]) - root_prefix_len - group_turn2_len, []) for i in idxs]
        if any(leaf.segment_len <= 0 for leaf in leaves):
            return None
        children.append((idxs, TreeNode(group_turn2_len, leaves)))

    return (root_prefix_len, children)
