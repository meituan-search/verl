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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from verl.utils.prefix_tree.tree import PrefixSubTrie, TrieNode

RangeSpec = tuple[int, int]

# Type alias for the boundary registry (see prepare_packed_label for the full WHY).
# Each entry is (boundary_flat_position, [(sample_idx, next_token), ...]).
# sample_idx is the index into the original samples list; next_token is the leaf's
# OWN next token at the boundary (int), NOT the owner's.
BoundaryRegistry = list[tuple[int, list[tuple[int, int]]]]


@dataclass
class PrefixTreeParams:
    """Metadata for a flattened PrefixTree batch."""

    prefix_range: RangeSpec
    leaf_ranges: list[RangeSpec]
    leaf_to_sample: list[int]
    sample_to_leaf_range: dict[int, RangeSpec]
    q_ranges: list[RangeSpec]
    k_ranges: list[RangeSpec]
    mask_types: list[str]
    total_seqlen_q: int
    total_seqlen_k: int
    tree_packed_tokens: Optional[Tensor] = None
    tree_packed_labels: Optional[Tensor] = None
    tree_packed_loss_mask: Optional[Tensor] = None
    tree_packed_position_ids: Optional[Tensor] = None

    # Boundary registry for the LCE boundary-patch fix (see prepare_packed_label).
    # Plain Python (list of tuples of ints) — serialises fine over RPC.
    # None when no boundaries exist (single-leaf subtries or no branching).
    boundary_registry: Optional[BoundaryRegistry] = None

    def __post_init__(self) -> None:
        if len(self.leaf_ranges) != len(self.leaf_to_sample):
            raise ValueError("leaf_ranges and leaf_to_sample must have the same length")
        if len(self.q_ranges) != len(self.k_ranges) or len(self.q_ranges) != len(self.mask_types):
            raise ValueError("q_ranges, k_ranges, and mask_types must have the same length")
        if set(self.leaf_to_sample) != set(self.sample_to_leaf_range):
            raise ValueError("sample_to_leaf_range must cover exactly the samples in leaf_to_sample")

        prefix_start, prefix_end = self.prefix_range
        if prefix_start != 0:
            raise ValueError("prefix_range must start at 0 in flattened PrefixTree layout")
        if prefix_end < prefix_start:
            raise ValueError("prefix_range must be non-decreasing")

        for leaf_range in self.leaf_ranges:
            leaf_start, leaf_end = leaf_range
            if leaf_end < leaf_start:
                raise ValueError("leaf ranges must be non-decreasing")

        if self.total_seqlen_q != self.total_seqlen_k:
            raise ValueError("PrefixTree expects matching q/k sequence lengths")
        if self.leaf_ranges and max(end for _, end in self.leaf_ranges) != self.total_seqlen_q:
            raise ValueError("max leaf range end must equal total sequence length")
        if not self.leaf_ranges and self.prefix_range[1] != self.total_seqlen_q:
            raise ValueError("prefix-only PrefixTree must end at total sequence length")

        for sample_idx, leaf_range in zip(self.leaf_to_sample, self.leaf_ranges, strict=False):
            if self.sample_to_leaf_range[sample_idx] != leaf_range:
                raise ValueError("sample_to_leaf_range does not match leaf_to_sample ordering")

        for name, tensor in {
            "tree_packed_tokens": self.tree_packed_tokens,
            "tree_packed_labels": self.tree_packed_labels,
            "tree_packed_loss_mask": self.tree_packed_loss_mask,
            "tree_packed_position_ids": self.tree_packed_position_ids,
        }.items():
            if tensor is not None and tensor.numel() != self.total_seqlen_q:
                raise ValueError(f"{name} must have total_seqlen_q elements")

    @property
    def prefix_len(self) -> int:
        return self.prefix_range[1] - self.prefix_range[0]

    @property
    def branch_lengths(self) -> list[int]:
        return [end - start for start, end in self.leaf_ranges]

    @property
    def num_samples(self) -> int:
        return len(self.leaf_to_sample)

    def get_leaf_range(self, sample_idx: int) -> RangeSpec:
        return self.sample_to_leaf_range[sample_idx]


__all__ = [
    "RangeSpec",
    "PrefixTreeParams",
    "BoundaryRegistry",
    "build_layout_from_tree_node",
    "prepare_packed_label",
]


def prepare_packed_label(
    samples: Sequence[Tensor],
    root_nodes: list[TrieNode],
    subtrie_valid_ids: set[int],
    leaf_node_id_to_samples: dict[int, list[int]],
) -> BoundaryRegistry:
    """Build the boundary registry for the LCE boundary-patch fix.

    WHY THIS EXISTS — the dedup → one-slot → LCE 1:1 → copy-to-all-leaves chain:
    -------------------------------------------------------------------------

    In the deduplicated flat layout, a shared ancestor's last token (the
    *boundary predictor position* — the position whose hidden state predicts
    the first divergent token of each leaf) occupies exactly ONE flat slot.
    The tree-packed labels place the OWNER's next-token id there (from
    ``rolled_samples[owner]``), and ``linear_cross_entropy`` (LCE) is strictly
    1:1 — one label per slot → one log-prob per slot.  So the flat logp at the
    boundary is ``log p(owner's next token | shared hidden)``.

    ``restore_flat_to_nested`` then reconstructs each sample's per-token logp
    by concatenating flat slices.  Every leaf sharing the ancestor reads the
    SAME ancestor slice (``flat_tensor[a:b]``), so the single boundary scalar
    is copied verbatim into every leaf's nested tensor.  Non-owner leaves
    receive the OWNER's next-token log-prob instead of their own — a
    ~19-nat error at every shared-segment junction.

    Donation (the previous fix) re-packed the boundary token per-leaf via
    layout offset surgery, but that broke the ``max leaf_range end ==
    total_seqlen_q`` invariant → 14% FA3 fallback.  This registry enables a
    cleaner fix: run LCE normally (owner's label at the boundary — temporarily
    "wrong" for non-owner leaves), then patch each leaf's boundary log-prob
    AFTER LCE with its OWN next-token log-prob computed from the shared hidden
    state (see ``post_processing_packed_lce`` in forward.py).

    BOUNDARY DETECTION:
    ------------------
    A flat position is a boundary iff its node has ≥2 children in the subtrie
    (≥2 descendant leaves diverge at that node's last token).  In a trie,
    children of the same node always start with distinct tokens, so ≥2
    children → ≥2 distinct next-tokens → the owner's next-token is wrong for
    non-owner leaves.  The boundary flat position is ``node._flat_end - 1``
    (the last token of the ancestor — the predictor for the divergent token).

    For each boundary, we collect every leaf in the subtree and its OWN next
    token: ``samples[leaf_sample][node._owner_offset + len(node.input_ids)]``
    — the token in the leaf's original sample right after the shared prefix.

    Args:
        samples: Original per-sample token tensors (1-D, variable length).
        root_nodes: Root nodes of the subtrie (from ``build_layout_from_tree_node``).
        subtrie_valid_ids: ``set`` of ``node_idx`` values for nodes in the subtrie
            (used to filter children).
        leaf_node_id_to_samples: Maps leaf node ``node_idx → [sample_idx, ...]``
            (representative first, then duplicate samples sharing the same leaf).

    Returns:
        ``list of (boundary_flat_pos, [(sample_idx, next_token), ...])``.
        Empty when no branching exists.  Plain Python — RPC-safe.
    """

    def _subtrie_children(node: TrieNode) -> list[TrieNode]:
        return [c for c in node.children.values() if c.node_idx in subtrie_valid_ids]

    def _collect_leaf_descendants(node: TrieNode) -> list[TrieNode]:
        """All leaf nodes (no children) in the subtree rooted at *node*."""
        result: list[TrieNode] = []
        for child in _subtrie_children(node):
            if not _subtrie_children(child):
                result.append(child)
            else:
                result.extend(_collect_leaf_descendants(child))
        return result

    registry: BoundaryRegistry = []

    # BFS walk to find branching nodes (≥2 children → boundary).
    for root in root_nodes:
        queue = [root]
        while queue:
            node = queue.pop(0)
            children = _subtrie_children(node)
            # Push children for BFS traversal regardless of boundary status.
            for c in children:
                queue.append(c)

            # Boundary condition: ≥2 children AND node has ≥1 token to emit.
            # A node with 0 tokens has no flat position (no predictor), and a
            # node with <2 children doesn't cause divergence here.
            if len(children) < 2 or len(node.input_ids) < 1:
                continue

            # The boundary is the LAST token of this shared ancestor.
            b_pos = node._flat_end - 1
            # The next-token position in the original sample: right after the
            # shared prefix ending at this node.  node._owner_offset is the
            # sample-local offset of this node's first token.
            next_token_pos = node._owner_offset + len(node.input_ids)

            leaves_info: list[tuple[int, int]] = []
            for leaf in _collect_leaf_descendants(node):
                # Expand to ALL samples sharing this leaf node (duplicates).
                for sample_idx in leaf_node_id_to_samples.get(leaf.node_idx, []):
                    # Zero-length leaves (empty response): the sample's last token IS
                    # this boundary, so next_token_pos is past the sample end — there is
                    # no response token here to predict, hence no boundary log-prob to
                    # patch for this leaf. Skip it (correct: nothing to fix for a leaf
                    # with no token at the boundary). Without this guard, indexing
                    # samples[sample_idx][next_token_pos] raises IndexError -> the whole
                    # micro-batch falls back to standard FA3 attention (losing the
                    # prefix-tree dedup + the boundary patch).
                    sample_len = samples[sample_idx].shape[0]
                    if next_token_pos >= sample_len:
                        continue
                    next_token = int(samples[sample_idx][next_token_pos].item())
                    leaves_info.append((sample_idx, next_token))

            if len(leaves_info) >= 2:
                registry.append((b_pos, leaves_info))

    return registry


def build_layout_from_tree_node(
    samples: Sequence[Tensor],
    subtrie: PrefixSubTrie,
    loss_masks_by_sample: Optional[Sequence[Tensor]] = None,
    position_ids_by_sample: Optional[Sequence[Tensor]] = None,
) -> PrefixTreeParams:
    """Build flat layout (PrefixTreeParams) from a PrefixSubTrie.

    Walks only the nodes in ``subtrie``, emitting tokens in BFS order
    (grouped by depth, with contiguous depth-level slices).
    Leaf ordering matches ``subtrie.leaf_to_sample``.

    Labels are computed per-sample (rolled) before packing: each node's labels
    come from ``rolled_samples[src][s:e]`` (next-token, last->0 EOS), so they are
    correct at prefix/leaf boundaries WITHOUT the donation mechanism.
    """

    valid_ids: set[int] = {n.node_idx for n in subtrie.nodes}
    # Clear stale layout attributes on ALL subtrie nodes before re-annotating.
    # TrieNode objects persist across micro-batches (shared via subtrie.node refs),
    # so _flat_start/_flat_end/_owner_* carry over from the PREVIOUS build_layout
    # call. Reset to None so any un-annotated node is detectable.
    for n in subtrie.nodes:
        n._flat_start = None
        n._flat_end = None
        n._owner_offset = None
    # Map node_idx → ordered list of sample_ids (first = representative, rest = zero-len duplicates)
    leaf_node_id_to_samples: dict[int, list[int]] = {}
    for nid, sid in zip(subtrie.leaf_node_ids, subtrie.leaf_to_sample, strict=False):
        leaf_node_id_to_samples.setdefault(nid, []).append(sid)
    leaf_node_id_to_sample: dict[int, int] = {nid: sids[0] for nid, sids in leaf_node_id_to_samples.items()}

    def _subtrie_children(node: TrieNode) -> list[TrieNode]:
        return [c for c in node.children.values() if c.node_idx in valid_ids]

    root_nodes = [n for n in subtrie.nodes if n.ancestor is None or n.ancestor.node_idx not in valid_ids]

    # Assign flat positions, build attention spec rectangles.
    q_ranges: list[RangeSpec] = []
    k_ranges: list[RangeSpec] = []
    mask_types: list[str] = []

    # Collect all subtrie nodes in BFS order.
    # No donation: each node emits its FULL input_ids (a branching parent keeps its
    # last token; the child does NOT re-emit it). The child attends to the parent's
    # shared last token via the FULL ancestor rectangle (q=child_range, k=parent_range),
    # so this is correct for standard attention — no RNN state carry needed.
    # Natural flat positions: every node's _flat_end - _flat_start == len(input_ids).
    bfs_order: list[TrieNode] = []
    for root in root_nodes:
        queue = [root]
        while queue:
            node = queue.pop(0)
            bfs_order.append(node)
            for child in _subtrie_children(node):
                queue.append(child)

    # Assign flat positions sequentially in BFS order (natural: no donation).
    pos = 0
    for node in bfs_order:
        node._flat_start = pos
        node._flat_end = pos + len(node.input_ids)
        pos = node._flat_end

    def _collect_descendants(node: TrieNode) -> list[TrieNode]:
        result: list[TrieNode] = []
        for child in _subtrie_children(node):
            result.append(child)
            result.extend(_collect_descendants(child))
        return result

    # Walk nodes in BFS order for attention rect emission (matches position layout).
    for node in bfs_order:
        if not node.input_ids or node._flat_start >= node._flat_end:
            # No tokens, or a zero-length duplicate leaf (a sample sharing a leaf
            # node with another sample): skip rects.
            continue
        # Mask range == layout range (natural positions, no donation).
        node_range: RangeSpec = (node._flat_start, node._flat_end)
        q_ranges.append(node_range)
        k_ranges.append(node_range)
        mask_types.append("causal")
        children = _subtrie_children(node)
        if not children:
            continue
        for desc in _collect_descendants(node):
            if not desc.input_ids:
                continue
            q_ranges.append((desc._flat_start, desc._flat_end))
            k_ranges.append(node_range)
            mask_types.append("full")

    # Build flat token layout.
    leaves_in_dfs: list[TrieNode] = []
    parent_of: dict[int, TrieNode] = {}

    def _annotate(node: TrieNode, owner_offset: int) -> int:
        node._owner_offset = owner_offset
        children = _subtrie_children(node)
        # A node is a "leaf" for layout purposes when the subtrie view maps some
        # sample to it (leaf_node_id_to_sample), NOT only when it is childless.
        # Under the greedy trie path a sample that is a strict prefix of another
        # terminates at an INTERNAL node (which has children); registering only
        # childless nodes dropped that sample -> segment_to_sample missed it ->
        # magi.py:304 "not covered by segment_to_sample". The segment path makes
        # every sample a childless leaf so this is a no-op there.
        node_sample = leaf_node_id_to_sample.get(node.node_idx)
        if not children:
            sample_idx = node_sample if node_sample is not None else leaf_node_id_to_sample[node.node_idx]
            node._owner_sample = sample_idx
            leaves_in_dfs.append(node)
            return sample_idx
        child_offset = owner_offset + len(node.input_ids)
        first_owner: Optional[int] = None
        for i, child in enumerate(children):
            parent_of[child.node_idx] = node
            owner = _annotate(child, child_offset)
            if i == 0:
                first_owner = owner
        # If this internal node is itself a sample's leaf (strict-prefix sample),
        # register it as a leaf so its sample is covered by segment_to_sample.
        if node_sample is not None:
            node._owner_sample = node_sample
            leaves_in_dfs.append(node)
            return node_sample
        node._owner_sample = first_owner
        return first_owner

    for root in root_nodes:
        _annotate(root, 0)

    device = samples[0].device
    flat_pieces: list[Tensor] = []
    flat_label_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []
    # Pre-roll each sample for labels: next-token prediction, last token -> 0 (EOS).
    # Per-sample roll (not global flat shift) makes labels correct at prefix/leaf
    # boundaries WITHOUT the donation mechanism.
    rolled_samples = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype, device=s.device)]) for s in samples]

    def _emit(node: TrieNode) -> None:
        if node.input_ids:
            children = _subtrie_children(node)
            s = node._owner_offset
            e = s + len(node.input_ids)
            if not children:
                leaf_sample = leaf_node_id_to_sample.get(node.node_idx)
                src = leaf_sample if leaf_sample is not None else node._owner_sample
            else:
                src = node._owner_sample
            if s < e:
                flat_pieces.append(samples[src][s:e])
                flat_label_pieces.append(rolled_samples[src][s:e])
                if flat_lm_pieces is not None:
                    flat_lm_pieces.append(loss_masks_by_sample[src][s:e])
                if flat_pid_pieces is not None:
                    flat_pid_pieces.append(position_ids_by_sample[src][s:e])
                else:
                    default_pid_pieces.append(torch.arange(s, e, device=device, dtype=torch.long))

    # Emit in BFS order (same order as position assignment).
    for node in bfs_order:
        _emit(node)

    tree_packed_tokens = (
        torch.cat(flat_pieces) if flat_pieces else torch.empty(0, dtype=samples[0].dtype, device=device)
    )
    tree_packed_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    # Labels: per-sample rolled (next-token), packed in BFS order to match packed logits.
    tree_packed_labels_tensor = (
        torch.cat(flat_label_pieces) if flat_label_pieces else torch.zeros_like(tree_packed_tokens)
    )
    if flat_pid_pieces is not None:
        tree_packed_position_ids = torch.cat(flat_pid_pieces)
    else:
        tree_packed_position_ids = (
            torch.cat(default_pid_pieces) if default_pid_pieces else torch.empty(0, dtype=torch.long, device=device)
        )

    # Build leaf ranges and ancestor chains in DFS order, interleaving zero-length
    # duplicate entries immediately after their representative so the last entry
    # (the last real leaf) still ends at total_seqlen_q (satisfying PrefixTreeParams).
    leaf_ranges: list[RangeSpec] = []
    leaf_to_sample_list: list[int] = []
    leaf_ancestor_ranges: list[list[RangeSpec]] = []

    # BFS positions mean leaves_in_dfs (DFS annotate order) may not be in flat order;
    # sort by _flat_start so leaf_ranges are in flat-position order (required by
    # PrefixTreeParams.__post_init__ and restore_flat_to_nested).
    leaves_sorted_by_flat = sorted(leaves_in_dfs, key=lambda l: l._flat_start)
    for leaf in leaves_sorted_by_flat:
        chain: list[RangeSpec] = []
        cur = parent_of.get(leaf.node_idx)
        while cur is not None:
            chain.append((cur._flat_start, cur._flat_end))
            cur = parent_of.get(cur.node_idx)
        chain.reverse()  # root first

        rep_range: RangeSpec = (leaf._flat_start, leaf._flat_end)
        sids = leaf_node_id_to_samples[leaf.node_idx]

        # Representative entry
        leaf_ranges.append(rep_range)
        leaf_to_sample_list.append(sids[0])
        leaf_ancestor_ranges.append(chain)

        # Zero-length entries for duplicates: ancestor chain extended with the rep's
        # leaf range so restore_flat_to_nested reconstructs the full sequence correctly.
        zero_range: RangeSpec = (rep_range[1], rep_range[1])
        for dup_sid in sids[1:]:
            leaf_ranges.append(zero_range)
            leaf_to_sample_list.append(dup_sid)
            leaf_ancestor_ranges.append(chain + [rep_range])

    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample_list, leaf_ranges, strict=False)}

    # prefix_range: the shared root segment (first root_node for single-prefix trees)
    prefix_range = (root_nodes[0]._flat_start, root_nodes[0]._flat_end)

    params = PrefixTreeParams(
        prefix_range=prefix_range,
        leaf_ranges=leaf_ranges,
        leaf_to_sample=leaf_to_sample_list,
        sample_to_leaf_range=sample_to_leaf_range,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        mask_types=mask_types,
        total_seqlen_q=tree_packed_tokens.numel(),
        total_seqlen_k=tree_packed_tokens.numel(),
        tree_packed_tokens=tree_packed_tokens,
        tree_packed_labels=tree_packed_labels_tensor,
        tree_packed_loss_mask=tree_packed_loss_mask,
        tree_packed_position_ids=tree_packed_position_ids,
        boundary_registry=prepare_packed_label(samples, root_nodes, valid_ids, leaf_node_id_to_samples),
    )
    params._leaf_ancestor_ranges = leaf_ancestor_ranges

    return params
