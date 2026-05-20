# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from verl.utils.prefix_tree_params import PrefixTreeParams, RangeSpec

__all__ = [
    'TreeNode',
    'build_prefix_tree_dense_mask',
    'build_prefix_tree_flex_spec',
    'build_multilevel_flex_spec',
    'build_prefix_tree_params',
    'extract_sample_tensor',
    'extract_sample_tensors',
    'longest_common_prefix_length',
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
    children: list['TreeNode'] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_prefix_tree_flex_spec(
    prefix_len: int,
    branch_lengths: Iterable[int],
) -> tuple[list[RangeSpec], list[RangeSpec], list[str]]:
    """Encode the root-shared PrefixTree mask as MAGI flex rectangles."""
    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative")

    q_ranges: list[RangeSpec] = [(0, prefix_len)]
    k_ranges: list[RangeSpec] = [(0, prefix_len)]
    mask_types = ['causal']

    current = prefix_len
    for branch_len in branch_lengths:
        if branch_len < 0:
            raise ValueError("branch_lengths must be non-negative")

        branch_range = (current, current + branch_len)
        q_ranges.append(branch_range)
        k_ranges.append((0, prefix_len))
        mask_types.append('full')

        q_ranges.append(branch_range)
        k_ranges.append(branch_range)
        mask_types.append('causal')
        current += branch_len

    return q_ranges, k_ranges, mask_types


def build_multilevel_flex_spec(
    root: 'TreeNode',
) -> tuple[list[RangeSpec], list[RangeSpec], list[str]]:
    """Encode a multi-level prefix tree as MAGI flex rectangles.

    The flat token layout is a DFS pre-order walk of the tree: each node's
    own tokens come before any of its descendants' tokens.

    For each node we emit:
    - One ``CAUSAL`` rectangle covering the node's own tokens (self-attention).
    - For each leaf descendant: one ``FULL`` rectangle from that leaf's tokens
      to this node's tokens (the leaf attends fully to every ancestor).

    This generalises the single-level encoding in ``build_prefix_tree_flex_spec``:
    a depth-1 tree with one root and B leaves produces identical output.

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
        mask_types.append('causal')

        if node.is_leaf:
            return

        # Every descendant attends fully to this node's tokens.
        for desc in _collect_descendants(node):
            if desc.segment_len == 0:
                continue
            desc_range: RangeSpec = (desc._flat_start, desc._flat_end)  # type: ignore[attr-defined]
            q_ranges.append(desc_range)
            k_ranges.append(node_range)
            mask_types.append('full')

        for child in node.children:
            _emit_node(child)

    _emit_node(root)

    return q_ranges, k_ranges, mask_types


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
    for (q_start, q_end), (k_start, k_end), mask_type in zip(q_ranges, k_ranges, mask_types):
        if q_start < 0 or k_start < 0 or q_end < q_start or k_end < k_start:
            raise ValueError("range specs must be non-decreasing and non-negative")
        if q_end > total_tokens or k_end > total_tokens:
            raise ValueError("range specs must not exceed total_tokens")
        if q_end == q_start or k_end == k_start:
            continue

        if mask_type == 'full':
            mask[q_start:q_end, k_start:k_end] = True
            continue
        if mask_type != 'causal':
            raise ValueError(f"Unsupported mask type: {mask_type}")

        q_pos = torch.arange(q_start, q_end, device=mask.device).unsqueeze(1)
        k_pos = torch.arange(k_start, k_end, device=mask.device).unsqueeze(0)
        mask[q_start:q_end, k_start:k_end] |= k_pos <= q_pos

    return mask


def longest_common_prefix_length(sequences: Sequence[Tensor]) -> int:
    """Return the longest common token prefix across 1D tensors."""
    reference = _validate_sequence_field('sequences', sequences)
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


def build_prefix_tree_params(
    tokens_by_sample: Sequence[Tensor],
    *,
    sample_indices: Optional[Sequence[int]] = None,
    prefix_len: Optional[int] = None,
    labels_by_sample: Optional[Sequence[Tensor]] = None,
    loss_masks_by_sample: Optional[Sequence[Tensor]] = None,
    position_ids_by_sample: Optional[Sequence[Tensor]] = None,
) -> PrefixTreeParams:
    """Build PrefixTreeParams for a root-shared batch flattened as [prefix][leaf_1][leaf_2]....

    `labels_by_sample` is only safe when the shared-prefix label slices are identical across all
    samples. Autoregressive next-token labels often diverge at the prefix/leaf boundary, so callers
    should omit labels unless they have already accounted for that ambiguity.
    """
    tokens_reference = _validate_sequence_field('tokens_by_sample', tokens_by_sample)
    normalized_sample_indices = _normalize_sample_indices(len(tokens_by_sample), sample_indices)

    if prefix_len is None:
        prefix_len = longest_common_prefix_length(tokens_by_sample)
    else:
        if prefix_len < 0:
            raise ValueError("prefix_len must be non-negative")
        if prefix_len > min(sequence.numel() for sequence in tokens_by_sample):
            raise ValueError("prefix_len cannot exceed the shortest sequence")
        _validate_shared_prefix('tokens_by_sample', tokens_by_sample, prefix_len)

    branch_lengths = [sequence.numel() - prefix_len for sequence in tokens_by_sample]
    prefix_range = (0, prefix_len)
    leaf_ranges = _build_leaf_ranges(prefix_len, branch_lengths)
    q_ranges, k_ranges, mask_types = build_prefix_tree_flex_spec(prefix_len, branch_lengths)

    flat_tokens = _flatten_root_shared_field('tokens_by_sample', tokens_by_sample, prefix_len)
    flat_labels = _flatten_optional_field('labels_by_sample', labels_by_sample, tokens_by_sample, prefix_len)
    flat_loss_mask = _flatten_optional_field(
        'loss_masks_by_sample', loss_masks_by_sample, tokens_by_sample, prefix_len
    )
    if position_ids_by_sample is None:
        flat_position_ids = _build_default_position_ids(prefix_len, branch_lengths, tokens_reference.device)
    else:
        flat_position_ids = _flatten_optional_field(
            'position_ids_by_sample', position_ids_by_sample, tokens_by_sample, prefix_len
        )

    return PrefixTreeParams(
        prefix_range=prefix_range,
        prefix_segments=[prefix_range],
        leaf_ranges=leaf_ranges,
        leaf_segments=list(leaf_ranges),
        leaf_to_sample=list(normalized_sample_indices),
        sample_to_leaf_range=dict(zip(normalized_sample_indices, leaf_ranges)),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        mask_types=mask_types,
        total_seqlen_q=flat_tokens.numel(),
        total_seqlen_k=flat_tokens.numel(),
        flat_tokens=flat_tokens,
        flat_labels=flat_labels,
        flat_loss_mask=flat_loss_mask,
        flat_position_ids=flat_position_ids,
    )


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


def _build_default_position_ids(
    prefix_len: int,
    branch_lengths: Sequence[int],
    device: torch.device,
) -> Tensor:
    pieces = [torch.arange(prefix_len, device=device, dtype=torch.long)]
    pieces.extend(
        torch.arange(prefix_len, prefix_len + branch_len, device=device, dtype=torch.long)
        for branch_len in branch_lengths
    )
    return torch.cat(pieces, dim=0)


def _build_leaf_ranges(prefix_len: int, branch_lengths: Sequence[int]) -> list[RangeSpec]:
    leaf_ranges: list[RangeSpec] = []
    current = prefix_len
    for branch_len in branch_lengths:
        leaf_range = (current, current + branch_len)
        leaf_ranges.append(leaf_range)
        current += branch_len
    return leaf_ranges


def _flatten_optional_field(
    name: str,
    field_by_sample: Optional[Sequence[Tensor]],
    tokens_by_sample: Sequence[Tensor],
    prefix_len: int,
) -> Optional[Tensor]:
    if field_by_sample is None:
        return None
    _validate_auxiliary_field(name, field_by_sample, tokens_by_sample)
    return _flatten_root_shared_field(name, field_by_sample, prefix_len)


def _flatten_root_shared_field(
    name: str,
    field_by_sample: Sequence[Tensor],
    prefix_len: int,
) -> Tensor:
    reference = _validate_sequence_field(name, field_by_sample)
    _validate_shared_prefix(name, field_by_sample, prefix_len)

    pieces = [reference[:prefix_len]]
    pieces.extend(sequence[prefix_len:] for sequence in field_by_sample)
    return torch.cat(pieces, dim=0)


def _normalize_sample_indices(num_samples: int, sample_indices: Optional[Sequence[int]]) -> list[int]:
    if sample_indices is None:
        return list(range(num_samples))
    if len(sample_indices) != num_samples:
        raise ValueError("sample_indices must have the same length as tokens_by_sample")
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices must be unique")
    return list(sample_indices)


def _validate_auxiliary_field(
    name: str,
    field_by_sample: Sequence[Tensor],
    tokens_by_sample: Sequence[Tensor],
) -> None:
    _validate_sequence_field(name, field_by_sample)
    if len(field_by_sample) != len(tokens_by_sample):
        raise ValueError(f"{name} must have the same length as tokens_by_sample")
    for index, (field_tensor, token_tensor) in enumerate(zip(field_by_sample, tokens_by_sample)):
        if field_tensor.numel() != token_tensor.numel():
            raise ValueError(
                f"{name}[{index}] must have the same number of elements as tokens_by_sample[{index}]"
            )


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


def _validate_shared_prefix(name: str, sequences: Sequence[Tensor], prefix_len: int) -> None:
    reference_prefix = sequences[0][:prefix_len]
    for index, sequence in enumerate(sequences[1:], start=1):
        if not torch.equal(sequence[:prefix_len], reference_prefix):
            raise ValueError(f"{name}[{index}] does not match the shared prefix")


def _validate_flat_tensor(flat_tensor: Tensor, prefix_tree_params: PrefixTreeParams) -> None:
    if flat_tensor.ndim == 0:
        raise ValueError("flat_tensor must have a sequence dimension")
    if flat_tensor.shape[0] != prefix_tree_params.total_seqlen_q:
        raise ValueError("flat_tensor sequence length must equal prefix_tree_params.total_seqlen_q")
