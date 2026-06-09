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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from verl.utils.prefix_tree.params import PrefixTreeParams, RangeSpec

__all__ = [
    "TreeNode",
    "build_prefix_tree_dense_mask",
    "build_prefix_tree_attention_spec",
    "build_layout_from_tree_node",
    "extract_sample_tensor",
    "extract_sample_tensors",
    "longest_common_prefix_length",
]


@dataclass
class TreeNode:
    """A node in a multi-level prefix tree.

    ``segment_len`` is the number of tokens *owned by this node* (i.e. the
    tokens that are new relative to its parent).  Leaf nodes have no children.

    Example — 3 samples each of length 1000 with 20% prefix at every level::

        root = TreeNode(200, [          # shared root prefix: tokens [0..200)
            TreeNode(200, [             # sub-prefix A:        tokens [200..400)
                TreeNode(600),          # leaf A1:             tokens [400..1000)
                TreeNode(600),          # leaf A2:             tokens [1000..1600)
            ]),
            TreeNode(200, [             # sub-prefix B:        tokens [1600..1800)
                TreeNode(600),          # leaf B1:             tokens [1800..2400)
            ]),
        ])
    """

    segment_len: int
    children: list[TreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_prefix_tree_attention_spec(
    root: TreeNode,
) -> tuple[list[RangeSpec], list[RangeSpec], list[str]]:
    """Encode a prefix tree as attention rectangle specs.

    The flat token layout is a DFS pre-order walk of the tree: each node's
    own tokens come before any of its descendants' tokens.

    For each node we emit:
    - One ``CAUSAL`` rectangle covering the node's own tokens (self-attention).
    - For each leaf descendant: one ``FULL`` rectangle from that leaf's tokens
      to this node's tokens (the leaf attends fully to every ancestor).

    Args:
        root: Root ``TreeNode`` of the tree.  ``segment_len`` must be > 0 for
            internal nodes; leaves may have ``segment_len`` == 0 (they are
            skipped).

    Returns:
        ``(q_ranges, k_ranges, mask_types)`` ready to pass to
        ``magi_attn_flex_key``.
    """
    q_ranges: list[RangeSpec] = []
    k_ranges: list[RangeSpec] = []
    mask_types: list[str] = []

    # Assign flat token offsets with a DFS pre-order walk.
    def _assign_offsets(node: TreeNode, start: int) -> int:
        node._flat_start = start  # type: ignore[attr-defined]
        node._flat_end = start + node.segment_len  # type: ignore[attr-defined]
        cur = node._flat_end  # type: ignore[attr-defined]
        for child in node.children:
            cur = _assign_offsets(child, cur)
        return cur

    _assign_offsets(root, 0)

    # Collect all descendant nodes (non-root) in DFS order.
    def _collect_descendants(node: TreeNode) -> list[TreeNode]:
        result: list[TreeNode] = []
        for child in node.children:
            result.append(child)
            result.extend(_collect_descendants(child))
        return result

    # For each node: emit its own CAUSAL self-rect, then one FULL rect per
    # descendant that attends to this node's tokens.
    def _emit_node(node: TreeNode) -> None:
        if node.segment_len == 0:
            return
        node_range: RangeSpec = (node._flat_start, node._flat_end)  # type: ignore[attr-defined]

        # Self-attention: causal over this node's own tokens.
        q_ranges.append(node_range)
        k_ranges.append(node_range)
        mask_types.append("causal")

        if node.is_leaf:
            return

        # Every descendant attends fully to this node's tokens.
        for desc in _collect_descendants(node):
            if desc.segment_len == 0:
                continue
            desc_range: RangeSpec = (desc._flat_start, desc._flat_end)  # type: ignore[attr-defined]
            q_ranges.append(desc_range)
            k_ranges.append(node_range)
            mask_types.append("full")

        for child in node.children:
            _emit_node(child)

    _emit_node(root)

    return q_ranges, k_ranges, mask_types


def build_layout_from_tree_node(
    samples: Sequence[Tensor],
    tree_root: TreeNode,
    leaf_to_sample: Sequence[int],
    loss_masks_by_sample: Optional[Sequence[Tensor]] = None,
    position_ids_by_sample: Optional[Sequence[Tensor]] = None,
) -> PrefixTreeParams:
    """Build flat layout (PrefixTreeParams) from a TreeNode spec.

    Consumed by the dynamic-trie detection path: ``build_tree_dynamic`` produces
    a ``(TreeNode, leaf_to_sample)`` pair describing the LOGICAL shared-prefix
    structure, and this function realises it as a PHYSICAL flat token layout
    plus q/k attention rectangles.

    For each node we track an "owner" sample (the first leaf descendant) and an
    "owner offset" (the cumulative segment_len of all ancestors). The owner's
    token slice at ``[offset, offset+segment_len)`` is what gets emitted into
    the flat tensor for that node.

    Args:
        samples: Per-sample 1-D token tensors. ``samples[i]`` is the original
            sequence for sample ``i``.
        tree_root: Root of the prefix tree. Its ``segment_len`` is the length
            of the shared root prefix.
        leaf_to_sample: For each leaf in DFS pre-order, the original sample
            index this leaf corresponds to. Length must equal the number of
            leaves in ``tree_root``.
        loss_masks_by_sample / position_ids_by_sample: optional per-sample
            auxiliary tensors. When None, position ids default to
            ``arange(owner_offset, owner_offset + segment_len)`` so each
            sample's RoPE positions match the original layout.

    Returns:
        PrefixTreeParams. Also stashes
        ``_leaf_ancestor_ranges`` on the params instance for downstream
        ``restore_flat_to_nested``.
    """
    # build_prefix_tree_attention_spec assigns _flat_start/_flat_end to every node
    # as a side effect — we rely on that for leaf_ranges + ancestor ranges below.
    q_ranges, k_ranges, mask_types = build_prefix_tree_attention_spec(tree_root)

    # Walk top-down: assign each node its owner sample (first leaf descendant)
    # and owner offset (= sum of ancestor segment_lens). Also collect leaves in
    # DFS order and a parent_of map for restoring ancestor chains.
    leaves_in_dfs: list[TreeNode] = []
    parent_of: dict[int, TreeNode] = {}
    leaf_cursor = [0]  # mutable closure cell

    def _annotate(node: TreeNode, owner_offset: int) -> int:
        node._owner_offset = owner_offset  # type: ignore[attr-defined]
        if node.is_leaf:
            sample_idx = int(leaf_to_sample[leaf_cursor[0]])
            leaf_cursor[0] += 1
            node._owner_sample = sample_idx  # type: ignore[attr-defined]
            leaves_in_dfs.append(node)
            return sample_idx
        child_offset = owner_offset + node.segment_len
        first_owner: Optional[int] = None
        for i, child in enumerate(node.children):
            parent_of[id(child)] = node
            owner = _annotate(child, child_offset)
            if i == 0:
                first_owner = owner
        node._owner_sample = first_owner  # type: ignore[attr-defined]
        return first_owner  # type: ignore[return-value]

    _annotate(tree_root, 0)

    # Emit flat tokens via DFS pre-order using each node's (owner_sample, owner_offset).
    device = samples[0].device
    flat_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []

    def _emit(node: TreeNode) -> None:
        if node.segment_len > 0:
            owner = node._owner_sample  # type: ignore[attr-defined]
            s = node._owner_offset  # type: ignore[attr-defined]
            e = s + node.segment_len
            flat_pieces.append(samples[owner][s:e])
            if flat_lm_pieces is not None:
                flat_lm_pieces.append(loss_masks_by_sample[owner][s:e])
            if flat_pid_pieces is not None:
                flat_pid_pieces.append(position_ids_by_sample[owner][s:e])
            else:
                default_pid_pieces.append(torch.arange(s, e, device=device, dtype=torch.long))
        for child in node.children:
            _emit(child)

    _emit(tree_root)

    flat_tokens = torch.cat(flat_pieces) if flat_pieces else torch.empty(0, dtype=samples[0].dtype, device=device)
    flat_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    if flat_pid_pieces is not None:
        flat_position_ids = torch.cat(flat_pid_pieces)
    else:
        flat_position_ids = (
            torch.cat(default_pid_pieces) if default_pid_pieces else torch.empty(0, dtype=torch.long, device=device)
        )

    leaf_ranges = [(leaf._flat_start, leaf._flat_end) for leaf in leaves_in_dfs]  # type: ignore[attr-defined]
    leaf_to_sample_list = [int(s) for s in leaf_to_sample]
    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample_list, leaf_ranges, strict=False)}

    leaf_ancestor_ranges: list[list[RangeSpec]] = []
    for leaf in leaves_in_dfs:
        chain: list[RangeSpec] = []
        cur = parent_of.get(id(leaf))
        while cur is not None:
            chain.append((cur._flat_start, cur._flat_end))  # type: ignore[attr-defined]
            cur = parent_of.get(id(cur))
        chain.reverse()  # root first
        leaf_ancestor_ranges.append(chain)

    prefix_range = (tree_root._flat_start, tree_root._flat_end)  # type: ignore[attr-defined]

    params = PrefixTreeParams(
        prefix_range=prefix_range,
        prefix_segments=[prefix_range],
        leaf_ranges=leaf_ranges,
        leaf_segments=list(leaf_ranges),
        leaf_to_sample=leaf_to_sample_list,
        sample_to_leaf_range=sample_to_leaf_range,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        mask_types=mask_types,
        total_seqlen_q=flat_tokens.numel(),
        total_seqlen_k=flat_tokens.numel(),
        flat_tokens=flat_tokens,
        flat_labels=None,
        flat_loss_mask=flat_loss_mask,
        flat_position_ids=flat_position_ids,
    )
    params._leaf_ancestor_ranges = leaf_ancestor_ranges  # type: ignore[attr-defined]
    return params


def build_prefix_tree_dense_mask(
    total_tokens: int,
    q_ranges: Sequence[RangeSpec],
    k_ranges: Sequence[RangeSpec],
    mask_types: Sequence[str],
    *,
    device: Optional[torch.device | str] = None,
) -> Tensor:
    """Materialize a dense mask from PrefixTree flex rectangles."""
    if len(q_ranges) != len(k_ranges) or len(q_ranges) != len(mask_types):
        raise ValueError("q_ranges, k_ranges, and mask_types must have the same length")
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")

    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
    for (q_start, q_end), (k_start, k_end), mask_type in zip(q_ranges, k_ranges, mask_types, strict=False):
        if q_start < 0 or k_start < 0 or q_end < q_start or k_end < k_start:
            raise ValueError("range specs must be non-decreasing and non-negative")
        if q_end > total_tokens or k_end > total_tokens:
            raise ValueError("range specs must not exceed total_tokens")
        if q_end == q_start or k_end == k_start:
            continue

        if mask_type == "full":
            mask[q_start:q_end, k_start:k_end] = True
            continue
        if mask_type != "causal":
            raise ValueError(f"Unsupported mask type: {mask_type}")

        q_pos = torch.arange(q_start, q_end, device=mask.device).unsqueeze(1)
        k_pos = torch.arange(k_start, k_end, device=mask.device).unsqueeze(0)
        mask[q_start:q_end, k_start:k_end] |= k_pos <= q_pos

    return mask


def longest_common_prefix_length(sequences: Sequence[Tensor]) -> int:
    """Return the longest common token prefix across 1D tensors."""
    reference = _validate_sequence_field("sequences", sequences)
    prefix_len = reference.numel()

    for sequence in sequences[1:]:
        prefix_len = min(prefix_len, sequence.numel())
        if prefix_len == 0:
            return 0

        mismatch = (reference[:prefix_len] != sequence[:prefix_len]).nonzero(as_tuple=False)
        if mismatch.numel() > 0:
            prefix_len = mismatch[0, 0].item()
        if prefix_len == 0:
            return 0

    return prefix_len


def extract_sample_tensor(
    flat_tensor: Tensor,
    prefix_tree_params: PrefixTreeParams,
    sample_idx: int,
) -> Tensor:
    """Restore one sample view by concatenating the shared prefix with its leaf segment."""
    _validate_flat_tensor(flat_tensor, prefix_tree_params)

    prefix_start, prefix_end = prefix_tree_params.prefix_range
    leaf_start, leaf_end = prefix_tree_params.get_leaf_range(sample_idx)
    prefix = flat_tensor[prefix_start:prefix_end]
    leaf = flat_tensor[leaf_start:leaf_end]
    return torch.cat([prefix, leaf], dim=0)


def extract_sample_tensors(
    flat_tensor: Tensor,
    prefix_tree_params: PrefixTreeParams,
) -> dict[int, Tensor]:
    """Restore per-sample views keyed by original sample index."""
    _validate_flat_tensor(flat_tensor, prefix_tree_params)
    return {
        sample_idx: extract_sample_tensor(flat_tensor, prefix_tree_params, sample_idx)
        for sample_idx in prefix_tree_params.leaf_to_sample
    }


def _validate_sequence_field(name: str, sequences: Sequence[Tensor]) -> Tensor:
    if not sequences:
        raise ValueError(f"{name} must contain at least one sequence")

    reference = sequences[0]
    if reference.ndim != 1:
        raise ValueError(f"{name} must contain 1D tensors")

    for sequence in sequences[1:]:
        if sequence.ndim != 1:
            raise ValueError(f"{name} must contain 1D tensors")
        if sequence.device != reference.device:
            raise ValueError(f"{name} must be on a single device")
        if sequence.dtype != reference.dtype:
            raise ValueError(f"{name} must have a single dtype")

    return reference


def _validate_flat_tensor(flat_tensor: Tensor, prefix_tree_params: PrefixTreeParams) -> None:
    if flat_tensor.ndim == 0:
        raise ValueError("flat_tensor must have a sequence dimension")
    if flat_tensor.shape[0] != prefix_tree_params.total_seqlen_q:
        raise ValueError("flat_tensor sequence length must equal prefix_tree_params.total_seqlen_q")
