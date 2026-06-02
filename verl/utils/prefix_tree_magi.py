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
"""Prefix-tree + MAGI utilities for verl SFT training.

Builds a MAGI attention key once per micro-batch from shared-prefix token
sequences, and provides helpers to restore flat outputs back to per-sample
NestedTensor form for downstream loss computation.

Usage (inside gptmodel_forward_model_engine):

    pt_batch = build_prefix_tree_micro_batch(model, input_ids, loss_mask, position_ids)
    if pt_batch is not None:
        output = model(
            input_ids=pt_batch.flat_input_ids,
            attention_mask=None,
            position_ids=pt_batch.flat_position_ids,
            packed_seq_params=None,
            magi_attention_key=pt_batch.magi_key,
        )
        output = restore_flat_to_nested(output, pt_batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor


@dataclass
class PrefixTreeMagiBatch:
    """Holds the flat layout and MAGI key for one prefix-tree micro-batch."""

    # flat input tensors ready to pass to model(...)
    flat_input_ids: Tensor  # (total_tokens,)
    flat_position_ids: Tensor  # (total_tokens,)
    flat_loss_mask: Optional[Tensor]  # (total_tokens,) or None

    # Attention keys — one will be None depending on prefix_tree_attention setting
    magi_key: object  # MAGI key (None when using flex)
    flex_key: object  # flex_attention block_mask (None when using magi)

    # mapping needed for output restoration
    # leaf_to_sample[i] = original sample index for leaf i
    leaf_to_sample: list[int]
    # leaf_ranges[i] = (start, end) token offset in flat layout for leaf i
    leaf_ranges: list[tuple[int, int]]
    prefix_range: tuple[int, int]

    # original batch size (= number of leaves for single-level tree)
    original_batch_size: int

    # number of real (non-padding) tokens; may be < flat_input_ids.shape[0]
    # when tp_size > 1 padding was added for sequence-parallel divisibility
    real_tokens: int = 0

    # leaf_ancestor_ranges[i] = list of (start,end) flat ranges that precede leaf i
    # For single-level: None (use prefix_range directly)
    # For multilevel: [(0, root_end), (turn2_start, turn2_end)] etc.
    leaf_ancestor_ranges: Optional[list[list[tuple[int, int]]]] = None

    # CP-local tensors: after magi dispatch, each CP rank only processes its assigned tokens.
    # When CP=1, these equal flat_input_ids/flat_position_ids/flat_loss_mask.
    # Shape: (local_tokens, ...) where local_tokens = total_tokens / cp_effective
    local_flat_input_ids: Optional[Tensor] = None
    local_flat_position_ids: Optional[Tensor] = None
    local_flat_loss_mask: Optional[Tensor] = None

    def __post_init__(self):
        if self.real_tokens == 0:
            self.real_tokens = int(self.flat_input_ids.shape[0])
        # Default local to full when not set (CP=1 or flex path)
        if self.local_flat_input_ids is None:
            self.local_flat_input_ids = self.flat_input_ids
        if self.local_flat_position_ids is None:
            self.local_flat_position_ids = self.flat_position_ids
        if self.local_flat_loss_mask is None:
            self.local_flat_loss_mask = self.flat_loss_mask


def build_prefix_tree_micro_batch(
    model,
    input_ids: NestedTensor,
    loss_mask: Optional[NestedTensor] = None,
    position_ids: Optional[NestedTensor] = None,
    prefix_segments_batch: Optional[list[list[tuple[int, int]]]] = None,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
) -> Optional[PrefixTreeMagiBatch]:
    """Build a PrefixTreeMagiBatch from a micro-batch of NestedTensor sequences.

    Returns None when there is no shared prefix (prefix_len == 0), signalling
    the caller to fall back to the standard attention path.

    Args:
        model: Megatron model (used to read num_heads / head_dim from config).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        loss_mask: Optional NestedTensor matching input_ids shape.
        position_ids: Optional NestedTensor matching input_ids shape.
            When None, default RoPE-compatible position IDs are generated.
        prefix_segments_batch: Optional per-sample prior knowledge injected by
            the dataset or trainer.  Each element is a list of
            ``(hash, cumulative_len)`` pairs (one per sub-turn) produced at
            data-load time.  When provided, prefix detection skips the O(batch ×
            seqlen) token comparison and uses the O(batch × turns) hash lookup
            instead.  Falls back to the scan when None or when no shared entry
            is found.

    Returns:
        PrefixTreeMagiBatch or None.
    """

    from verl.utils.prefix_tree_utils import (
        build_prefix_tree_params,
        longest_common_prefix_length,
    )

    # Unpack NestedTensor into a list of 1-D CPU tensors for prefix detection
    offsets = input_ids.offsets()  # (batch_size+1,)
    lengths = offsets.diff().tolist()
    flat_vals = input_ids.values()  # (total_nnz,)

    tokens_by_sample: list[Tensor] = []
    pos = 0
    for length in lengths:
        tokens_by_sample.append(flat_vals[pos : pos + int(length)])
        pos += int(length)

    # Fast path: use injected prior knowledge when available.
    import os as _os

    if prefix_segments_batch is not None and len(prefix_segments_batch) == len(tokens_by_sample):
        prefix_len = _resolve_prefix_len_from_segments(prefix_segments_batch)
        if _os.environ.get("DEBUG_PREFIX_LEN") == "1":
            scan_len = longest_common_prefix_length(tokens_by_sample)
            T = tokens_by_sample[0].shape[0] if tokens_by_sample else 0
            print(
                f"[PREFIX_LEN] seg_len={prefix_len} scan_len={scan_len} T={T} "
                f"n_segs={[len(s) for s in prefix_segments_batch]}",
                flush=True,
            )
    else:
        prefix_len = longest_common_prefix_length(tokens_by_sample)

    if prefix_len == 0:
        return None

    # Optionally unpack loss_mask and position_ids per sample
    loss_masks_by_sample: Optional[list[Tensor]] = None
    if loss_mask is not None:
        lm_vals = loss_mask.values()
        lm_offsets = loss_mask.offsets()
        lm_lengths = lm_offsets.diff().tolist()
        loss_masks_by_sample = []
        pos = 0
        for length in lm_lengths:
            loss_masks_by_sample.append(lm_vals[pos : pos + int(length)])
            pos += int(length)

    position_ids_by_sample: Optional[list[Tensor]] = None
    if position_ids is not None:
        pid_vals = position_ids.values()
        pid_offsets = position_ids.offsets()
        pid_lengths = pid_offsets.diff().tolist()
        position_ids_by_sample = []
        pos = 0
        for length in pid_lengths:
            position_ids_by_sample.append(pid_vals[pos : pos + int(length)])
            pos += int(length)

    # Try multi-level tree if prefix_segments are available.
    # Use token-scan root_len (not segment-based) for correct boundary alignment.
    multilevel_result = None
    if prefix_segments_batch is not None:
        actual_root_len = longest_common_prefix_length(tokens_by_sample)
        if actual_root_len > 0:
            multilevel_result = _resolve_multilevel_tree(tokens_by_sample, prefix_segments_batch, actual_root_len)

    if multilevel_result is not None:
        root_len, children_info = multilevel_result
        params = _build_multilevel_prefix_tree_params(
            tokens_by_sample,
            root_len,
            children_info,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )
    else:
        params = build_prefix_tree_params(
            tokens_by_sample,
            prefix_len=prefix_len,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )

    # Sequence parallel (TP>1) requires total_tokens % align_size == 0.
    # When CP>1, FA3 uses align_size = tp_size * cp_size * 2 (interleaved CP chunks).
    # We must match this so MAGI and FA3 process identical token counts.
    real_tokens = params.flat_tokens.shape[0]
    if tp_size > 1:
        align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
        total = real_tokens
        pad_len = (align_size - total % align_size) % align_size
        if pad_len > 0:
            import torch as _torch

            params.flat_tokens = _torch.cat([params.flat_tokens, params.flat_tokens.new_zeros(pad_len)])
            params.flat_position_ids = _torch.cat(
                [params.flat_position_ids, params.flat_position_ids.new_zeros(pad_len)]
            )
            if params.flat_loss_mask is not None:
                params.flat_loss_mask = _torch.cat([params.flat_loss_mask, params.flat_loss_mask.new_zeros(pad_len)])
            params.total_seqlen_q += pad_len
            params.total_seqlen_k += pad_len
            # Do NOT add rectangles for padding tokens — they are stripped before loss.
            # MAGI assigns zero attention weight to out-of-range positions automatically.

    # Build attention key — only the requested type
    if attention_type == "magi":
        magi_key = _build_magi_key(model, params)
        flex_key = None
    else:  # flex (default)
        device = params.flat_tokens.device
        flex_key = _build_flex_key(params, device)
        magi_key = None

    # Note: CP dispatch at embedding level (dispatch before model, undispatch after)
    # is the correct MAGI integration pattern but requires wrapping the full model
    # forward including FFN layers. This is a TODO for proper CP support.
    # For now, local_* == flat_* and dispatch/undispatch remain inside _magi_attn_forward.
    local_flat_tokens = params.flat_tokens
    local_flat_position_ids = params.flat_position_ids
    local_flat_loss_mask = params.flat_loss_mask

    return PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=params.flat_loss_mask,
        magi_key=magi_key,
        flex_key=flex_key,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=params.num_samples,
        real_tokens=real_tokens,
        leaf_ancestor_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        local_flat_input_ids=local_flat_tokens,
        local_flat_position_ids=local_flat_position_ids,
        local_flat_loss_mask=local_flat_loss_mask,
    )


def restore_flat_to_nested(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
) -> NestedTensor:
    """Restore a flat (total_tokens, ...) tensor to a per-sample NestedTensor.

    Each sample's view is ``[prefix_tokens || leaf_tokens]`` concatenated,
    matching the original per-sample sequence length.

    Args:
        flat_tensor: Tensor with first dimension == total_tokens.
        pt_batch: PrefixTreeMagiBatch from build_prefix_tree_micro_batch.

    Returns:
        NestedTensor of shape (batch_size, variable_seqlen, ...).
    """
    prefix_start, prefix_end = pt_batch.prefix_range
    prefix_slice = flat_tensor[prefix_start:prefix_end]

    # Reconstruct per-sample tensors in original sample order
    # leaf_to_sample gives the original sample index for each leaf
    n = pt_batch.original_batch_size
    sample_tensors: list[Optional[Tensor]] = [None] * n

    for leaf_idx, sample_idx in enumerate(pt_batch.leaf_to_sample):
        leaf_start, leaf_end = pt_batch.leaf_ranges[leaf_idx]
        leaf_slice = flat_tensor[leaf_start:leaf_end]
        if pt_batch.leaf_ancestor_ranges is not None:
            # Multilevel: concatenate all ancestor ranges + leaf
            parts = [flat_tensor[s:e] for s, e in pt_batch.leaf_ancestor_ranges[leaf_idx]]
            parts.append(leaf_slice)
            sample_tensors[sample_idx] = torch.cat(parts, dim=0)
        else:
            sample_tensors[sample_idx] = torch.cat([prefix_slice, leaf_slice], dim=0)

    assert all(t is not None for t in sample_tensors), (
        "restore_flat_to_nested: some sample indices were not covered by leaf_to_sample"
    )

    # Use as_nested_tensor (not nested_tensor) to preserve grad_fn through the cat ops.
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _resolve_prefix_len_from_segments(
    prefix_segments_batch: list[list[tuple[int, int]]],
) -> int:
    """Return the longest shared-prefix length derivable from per-sample segment lists.

    Each element of *prefix_segments_batch* is a list of ``(hash, cumulative_len)``
    pairs produced by the dataset at load time.  Two samples share turn k when all
    samples have the same per-turn hash at position k.

    Algorithm: walk turn-by-turn; stop at the first turn where hashes diverge.
    Return the cumulative_len from sample 0 at the last shared turn (0 if none).

    Using only hashes (not cum_len) for comparison because tokenization boundary
    effects can shift cum_len slightly between samples even for identical turns.
    """
    n = len(prefix_segments_batch)
    if n == 0:
        return 0

    # Find the minimum number of turns across all samples
    min_turns = min(len(segs) for segs in prefix_segments_batch)
    if min_turns == 0:
        return 0

    best = 0
    for turn_idx in range(min_turns):
        # Check if all samples have the same hash at this turn position
        h0 = prefix_segments_batch[0][turn_idx][0]
        if all(prefix_segments_batch[i][turn_idx][0] == h0 for i in range(1, n)):
            # All match — use cum_len from sample 0 as the shared prefix boundary
            best = prefix_segments_batch[0][turn_idx][1]
        else:
            break  # first divergence — stop

    return best


def _resolve_multilevel_tree(
    tokens_by_sample: list,
    prefix_segments_batch: list[list[tuple[int, int]]],
    root_prefix_len: int,
) -> Optional[object]:
    """Detect a 2-level tree structure from prefix_segments and return a TreeNode, or None.

    Groups samples by their first post-root segment hash. If ≥2 groups each
    have ≥2 samples, a 2-level tree exists and we return a TreeNode root.
    """
    from collections import defaultdict

    from verl.utils.prefix_tree_utils import TreeNode

    n = len(tokens_by_sample)
    if prefix_segments_batch is None or n < 4:
        return None

    # Group samples by the hash of their first post-root turn.
    # With per-turn hashes in prefix_segments, use the hash directly
    # (no token scan needed — O(batch×turns) instead of O(batch×seqlen)).
    groups: dict[int, list[int]] = defaultdict(list)
    for i, segs in enumerate(prefix_segments_batch):
        # Find first segment whose cumlen > root_prefix_len — this is the first post-root turn
        next_seg = next((s for s in segs if s[1] > root_prefix_len), None)
        if next_seg is None:
            return None  # no post-root turns
        # Use per-turn hash directly (hash of just that turn's tokens)
        groups[next_seg[0]].append(i)

    # Need ≥2 groups each with ≥2 samples for multi-level to be worthwhile
    useful = [(h, idxs) for h, idxs in groups.items() if len(idxs) >= 2]
    if len(useful) < 2:
        return None

    # Use token scan (not segment hashes) to get exact turn2 shared prefix length per group.
    # Segment hashes can mis-align due to chat-template boundary effects.
    from verl.utils.prefix_tree_utils import longest_common_prefix_length

    children = []
    for _h, idxs in useful:
        group_tokens = [tokens_by_sample[i] for i in idxs]
        # scan from root_prefix_len onwards for shared content within this group
        suffixes = [t[root_prefix_len:] for t in group_tokens]
        shared_suffix_len = longest_common_prefix_length(suffixes)
        if shared_suffix_len <= 0:
            return None
        group_turn2_len = shared_suffix_len
        leaves = [TreeNode(int(tokens_by_sample[i].shape[0]) - root_prefix_len - group_turn2_len, []) for i in idxs]
        if any(leaf.segment_len <= 0 for leaf in leaves):
            return None
        children.append((idxs, TreeNode(group_turn2_len, leaves)))

    return (root_prefix_len, children)  # (root_len, [(sample_idxs, child_node), ...])


def _build_multilevel_prefix_tree_params(
    tokens_by_sample: list,
    root_len: int,
    children_info: list,  # [(sample_idxs, child_TreeNode), ...]
    loss_masks_by_sample=None,
    position_ids_by_sample=None,
):
    """Build PrefixTreeParams for a 2-level tree using build_multilevel_flex_spec."""
    import torch

    from verl.utils.prefix_tree_params import PrefixTreeParams
    from verl.utils.prefix_tree_utils import TreeNode, build_multilevel_flex_spec

    # Build TreeNode for flex_spec
    tree_children = [child_node for _, child_node in children_info]
    root_node = TreeNode(root_len, tree_children)

    # Assign flat offsets via DFS pre-order (mirrors build_multilevel_flex_spec internals)
    def _assign_offsets(node, start):
        node._flat_start = start
        node._flat_end = start + node.segment_len
        cur = node._flat_end
        for child in node.children:
            cur = _assign_offsets(child, cur)
        return cur

    total_tokens = _assign_offsets(root_node, 0)

    # Build flat token tensor: root + each child group + each leaf in DFS order
    device = tokens_by_sample[0].device
    flat_parts = [tokens_by_sample[0][:root_len]]  # shared root (same across all)

    leaf_ranges = []
    leaf_to_sample = []

    def _flatten_node(node, sample_idxs_list, depth):
        if node.is_leaf:
            # leaf: take this sample's unique tail
            sample_idx = sample_idxs_list[0]
            tok = tokens_by_sample[sample_idx]
            # tail = everything after root + all ancestor segments
            tail_start = node._flat_start
            tail_end = node._flat_end
            # The sample tokens after root: tok[root_len:]
            # We need to locate this leaf's content in the sample
            # In DFS pre-order: node._flat_start = root_len + sum of prior sibling/parent segments
            # The sample's tokens[root_len + parent_len : ] = leaf content
            # parent_len = parent node's turn2 segment = child_node.segment_len for that branch
            leaf_ranges.append((tail_start, tail_end))
            leaf_to_sample.append(sample_idx)
            return
        # internal node: add this node's own tokens (shared by its group)
        sample_idx = sample_idxs_list[0]
        tok = tokens_by_sample[sample_idx]
        # Compute where this node's tokens start in the original sample
        # For root: sample[0:root_len]
        # For depth-1 child: sample[root_len : root_len + child.segment_len]
        if depth == 0:
            content = tok[: node._flat_end]  # root tokens
        else:
            content = tok[root_len : root_len + node.segment_len]  # turn2 tokens
        flat_parts.append(content)

        # Recurse into children, distributing sample indices
        leaf_idx = 0
        for child in node.children:
            n_leaves = _count_leaves(child)
            _flatten_node(child, sample_idxs_list[leaf_idx : leaf_idx + n_leaves], depth + 1)
            leaf_idx += n_leaves

    def _count_leaves(node):
        if node.is_leaf:
            return 1
        return sum(_count_leaves(c) for c in node.children)

    # Flatten: root already added; now children in order
    flat_parts = [tokens_by_sample[0][:root_len]]  # root (same for all samples)
    leaf_ranges = []
    leaf_to_sample = []
    leaf_ancestor_ranges = []  # [(root_range, turn2_range), ...] per leaf

    cur_offset = root_len
    for child_idxs, child_node in children_info:
        # Add turn2 tokens (from first sample of this group)
        sample0 = child_idxs[0]
        turn2_tokens = tokens_by_sample[sample0][root_len : root_len + child_node.segment_len]
        flat_parts.append(turn2_tokens)
        turn2_flat_start = cur_offset
        cur_offset += child_node.segment_len
        turn2_flat_range = (turn2_flat_start, cur_offset)

        # Add each leaf
        for leaf_node, sample_idx in zip(child_node.children, child_idxs, strict=False):
            leaf_start = cur_offset
            leaf_len = leaf_node.segment_len
            leaf_tokens = tokens_by_sample[sample_idx][root_len + child_node.segment_len :]
            assert leaf_tokens.shape[0] == leaf_len, f"leaf token mismatch: {leaf_tokens.shape[0]} != {leaf_len}"
            flat_parts.append(leaf_tokens)
            leaf_ranges.append((leaf_start, leaf_start + leaf_len))
            leaf_to_sample.append(sample_idx)
            leaf_ancestor_ranges.append([(0, root_len), turn2_flat_range])
            cur_offset += leaf_len

    flat_tokens = torch.cat(flat_parts)
    assert flat_tokens.shape[0] == total_tokens, f"flat token count mismatch: {flat_tokens.shape[0]} != {total_tokens}"

    # Build loss mask
    flat_loss_mask = None
    if loss_masks_by_sample is not None:
        lm_parts = [loss_masks_by_sample[0][:root_len]]
        for child_idxs, child_node in children_info:
            sample0 = child_idxs[0]
            lm_parts.append(loss_masks_by_sample[sample0][root_len : root_len + child_node.segment_len])
            for leaf_node, sample_idx in zip(child_node.children, child_idxs, strict=False):
                lm_parts.append(loss_masks_by_sample[sample_idx][root_len + child_node.segment_len :])
        flat_loss_mask = torch.cat(lm_parts)

    # Build position ids: root is 0..root_len-1, each turn2 starts at root_len,
    # each leaf starts at root_len + turn2_len (matching the original sample positions)
    # This matches what the model expects for RoPE: positions within each sample are
    # relative to the start of that sample's content.
    pid_parts = [torch.arange(root_len, device=device)]  # root: 0..root_len-1
    for child_idxs, child_node in children_info:
        # turn2 positions: root_len .. root_len + turn2_len - 1
        pid_parts.append(torch.arange(root_len, root_len + child_node.segment_len, device=device))
        for leaf_node, _sample_idx in zip(child_node.children, child_idxs, strict=False):
            # leaf positions: root_len + turn2_len .. root_len + turn2_len + leaf_len - 1
            leaf_start_pos = root_len + child_node.segment_len
            pid_parts.append(torch.arange(leaf_start_pos, leaf_start_pos + leaf_node.segment_len, device=device))
    flat_position_ids = torch.cat(pid_parts)

    # Get attention rectangles from multilevel spec
    q_ranges, k_ranges, mask_types = build_multilevel_flex_spec(root_node)

    params = PrefixTreeParams(
        prefix_range=(0, root_len),
        prefix_segments=[(0, root_len)],
        leaf_ranges=leaf_ranges,
        leaf_segments=leaf_ranges,
        leaf_to_sample=leaf_to_sample,
        sample_to_leaf_range=dict(zip(leaf_to_sample, leaf_ranges, strict=False)),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        mask_types=mask_types,
        total_seqlen_q=total_tokens,
        total_seqlen_k=total_tokens,
        flat_tokens=flat_tokens,
        flat_loss_mask=flat_loss_mask,
        flat_position_ids=flat_position_ids,
        multilevel=True,
    )
    params._leaf_ancestor_ranges = leaf_ancestor_ranges  # attach for PrefixTreeMagiBatch
    return params


def build_prefix_segments_single_turn(
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> list[tuple[int, int]]:
    """Build a single-entry prefix_segments list for one sample.

    Used by RL trainers when per-sub-turn boundaries are unavailable.
    The single entry covers the entire real (non-pad) prompt.

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


def _build_flex_key(params, device):
    """Build a torch flex_attention block_mask from PrefixTreeParams.

    The mask encodes the prefix-tree attention pattern:
    - Prefix tokens: causal self-attention
    - Leaf tokens: full attention to prefix + causal self-attention within same leaf
    - Cross-leaf attention: blocked (leaf_i cannot see leaf_j)

    Returns a compiled block_mask usable with torch.nn.attention.flex_attention.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    # Build lookup arrays on CPU then move to device
    total = params.total_seqlen_q
    prefix_end = params.prefix_range[1]  # == prefix_len

    # leaf_id[t] = which leaf token t belongs to (-1 = prefix)
    import torch as _torch

    leaf_id = _torch.full((total,), -1, dtype=_torch.int32)
    for i, (s, e) in enumerate(params.leaf_ranges):
        leaf_id[s:e] = i
    leaf_id = leaf_id.to(device)

    def prefix_tree_mask(b, h, q_idx, kv_idx):
        # Within prefix: causal
        # Leaf q attending to prefix k: always allowed
        # Leaf q attending to same-leaf k: causal
        # Leaf q attending to different-leaf k: blocked
        q_leaf = leaf_id[q_idx]
        k_leaf = leaf_id[kv_idx]
        in_prefix_k = kv_idx < prefix_end
        same_leaf = (q_leaf == k_leaf) & (q_leaf >= 0)
        causal = kv_idx <= q_idx
        return (in_prefix_k & causal) | (same_leaf & causal) | (in_prefix_k & (q_leaf >= 0))

    # _compile=False: avoid Triton JIT which takes minutes for new shapes.
    # Memory is handled at the call site via torch.utils.checkpoint.
    block_mask = create_block_mask(
        prefix_tree_mask, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device, _compile=False
    )
    block_mask._leaf_id = leaf_id  # keep closure alive
    return block_mask


def _build_magi_key(model, params):
    """Construct a magi_attn_flex_key from PrefixTreeParams and model config."""
    import torch.distributed as dist
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    from verl.utils.megatron_utils import unwrap_model

    cfg = unwrap_model(model).config
    num_heads_q = cfg.num_attention_heads
    # GQA: num_query_groups may be set; fall back to num_heads_q if not
    num_heads_kv = getattr(cfg, "num_query_groups", num_heads_q) or num_heads_q
    head_dim = cfg.kv_channels  # hidden_size // num_attention_heads

    # Use the CP group so MAGI dispatch operates across CP ranks only.
    # The key uses full T token indices (not SP-scattered) since MAGI's dispatch
    # understands the logical token layout regardless of SP's physical distribution.
    # NOTE: CP>1 correctness requires dispatch to operate on full T tokens before
    # SP scatter — this is TODO. Currently dispatch/undispatch inside _magi_attn_forward
    # operate on SP-scattered T/TP tokens which is incorrect for CP>1.
    try:
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = dist.group.WORLD

    magi_key = magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(params.q_ranges),
        k_ranges=AttnRanges.from_ranges(params.k_ranges),
        attn_mask_type=[AttnMaskType(m) for m in params.mask_types],
        total_seqlen_q=params.total_seqlen_q,
        total_seqlen_k=params.total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )
    return magi_key


def _build_magi_key_sp_scaled(original_key, model, tp_size: int):
    """Rebuild a MAGI key scaled for the SP-scattered token domain (T/TP seqlen).

    With sequence_parallel=True, the embedding scatter gives T/TP tokens per TP rank.
    The MAGI key must use T/TP as total_seqlen so dispatch/undispatch in
    _magi_attn_forward operate on T/TP tokens → T/(TP*CP) per rank → back to T/TP.

    This function creates a new key with all token offsets divided by tp_size.
    """
    import torch.distributed as dist
    from magi_attention.api import DistAttnConfig, magi_attn_flex_key
    from magi_attention.common import AttnRanges
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig

    try:
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = dist.group.WORLD

    q_ranges_sp = [(r.start // tp_size, r.end // tp_size) for r in original_key.q_ranges]
    k_ranges_sp = [(r.start // tp_size, r.end // tp_size) for r in original_key.k_ranges]
    total_q_sp = original_key.total_seqlen_q // tp_size
    total_k_sp = original_key.total_seqlen_k // tp_size

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(q_ranges_sp),
        k_ranges=AttnRanges.from_ranges(k_ranges_sp),
        attn_mask_type=list(original_key.attn_mask_type),
        total_seqlen_q=total_q_sp,
        total_seqlen_k=total_k_sp,
        num_heads_q=original_key.num_heads_q,
        num_heads_kv=original_key.num_heads_kv,
        head_dim=original_key.head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )
