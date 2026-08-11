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

# Boundary registry: (boundary_flat_position, [(sample_idx, next_token), ...]).
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
    flat_end: dict[int, int],
    owner_offset: dict[int, int],
) -> BoundaryRegistry:
    """Build boundary registry for LCE boundary-patch: maps flat positions with ≥2 branching children to per-leaf
    (sample_idx, next_token) pairs, so restore_flat_to_nested can patch non-owner leaf boundary log-probs after LCE."""

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
            b_pos = flat_end[node.node_idx] - 1
            next_token_pos = owner_offset[node.node_idx] + len(node.input_ids)

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
    """Build flat layout (PrefixTreeParams) from a PrefixSubTrie via single BFS pass."""
    # Map node_idx → ordered list of sample_ids (first = representative, rest = duplicates)
    leaf_node_id_to_samples: dict[int, list[int]] = {}
    for nid, sid in zip(subtrie.leaf_node_ids, subtrie.leaf_to_sample, strict=False):
        leaf_node_id_to_samples.setdefault(nid, []).append(sid)
    leaf_node_id_to_sample: dict[int, int] = {nid: sids[0] for nid, sids in leaf_node_id_to_samples.items()}
    # seq_id → leaf node_idx for fallback owner lookup.
    seq_id_to_leaf_node_idx: dict[int, int] = {}
    for nid, sids in leaf_node_id_to_samples.items():
        for sid in sids:
            seq_id_to_leaf_node_idx[sid] = nid

    valid_ids: set[int] = {n.node_idx for n in subtrie.nodes}

    root_nodes = [n for n in subtrie.nodes if n.ancestor is None or n.ancestor.node_idx not in valid_ids]
    device = samples[0].device
    rolled_samples = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype, device=s.device)]) for s in samples]

    # Local layout dicts — no TrieNode mutation.
    flat_start: dict[int, int] = {}
    flat_end: dict[int, int] = {}
    owner_offset: dict[int, int] = {}
    owner_sample: dict[int, int | None] = {}

    # Single BFS: positions, owner annotation, pack tokens/labels/masks.
    q_ranges: list[RangeSpec] = []
    k_ranges: list[RangeSpec] = []
    mask_types: list[str] = []
    flat_pieces: list[Tensor] = []
    flat_label_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []
    bfs_order: list[TrieNode] = []

    pos = 0
    for root in root_nodes:
        # Compute root's sample-local offset: sum of global ancestor lengths
        # (ancestors not in this subtrie that precede the root in the sample).
        root_offset = 0
        anc = root.ancestor
        while anc is not None and anc.input_ids is not None:
            root_offset += len(anc.input_ids)
            anc = anc.ancestor
        # Queue items: (node, parent_owner_offset)
        queue: list[tuple[TrieNode, int]] = [(root, root_offset)]
        while queue:
            node, parent_offset = queue.pop(0)
            bfs_order.append(node)
            nid = node.node_idx
            # Only queue subtrie children.
            children = [c for c in node.children.values() if c.node_idx in valid_ids]

            flat_start[nid] = pos
            flat_end[nid] = pos + len(node.input_ids)
            # Sample-local offset passed from parent (descendant direction).
            owner_offset[nid] = parent_offset

            node_sample = leaf_node_id_to_sample.get(nid)
            if not children:
                owner = node_sample
                if owner is None:
                    # Subtrie leaf that's internal in global trie: use any
                    # sequence_id to find a leaf node_idx known to the subtrie.
                    for seq_id in node.sequence_ids:
                        leaf_nid = seq_id_to_leaf_node_idx.get(seq_id)
                        if leaf_nid is not None:
                            owner = leaf_node_id_to_sample[leaf_nid]
                            break
                owner_sample[nid] = owner
                if owner is not None:
                    pass  # owner tracked in dict, leaf ranges built from subtrie.leaf_node_ids
            elif node_sample is not None:
                owner_sample[nid] = node_sample

            child_offset = parent_offset + len(node.input_ids)
            for child in children:
                queue.append((child, child_offset))
            pos = flat_end[nid]

    # Post-BFS: set owner for internal nodes (first child's owner propagates up).
    for node in reversed(bfs_order):
        nid = node.node_idx
        if nid in owner_sample and owner_sample[nid] is not None:
            continue
        # Walk global descendants to find any subtrie leaf with a known sample.
        stack = list(node.children.values())
        while stack:
            c = stack.pop()
            cid = c.node_idx
            if cid in owner_sample and owner_sample[cid] is not None:
                owner_sample[nid] = owner_sample[cid]
                break
            stack.extend(c.children.values())

    # Pack tokens/labels/masks/position_ids (second pass, now owners are set).
    for node in bfs_order:
        if len(node.input_ids) == 0:
            continue
        nid = node.node_idx
        s = owner_offset[nid]
        e = s + len(node.input_ids)
        if s < e:
            src = owner_sample.get(nid, 0)
            if src is None:
                continue  # pruned node, not owned by any sample in this shard
            flat_pieces.append(samples[src][s:e])
            flat_label_pieces.append(rolled_samples[src][s:e])
            if flat_lm_pieces is not None:
                flat_lm_pieces.append(loss_masks_by_sample[src][s:e])

            # Attention rectangles + position IDs.
            fs, fe = flat_start[nid], flat_end[nid]
            if flat_pid_pieces is not None:
                flat_pid_pieces.append(position_ids_by_sample[src][s:e])
            else:
                default_pid_pieces.append(torch.arange(s, e, device=device, dtype=torch.long))

            node_range: RangeSpec = (fs, fe)
            q_ranges.append(node_range)
            k_ranges.append(node_range)
            mask_types.append("causal")
            node_children = list(node.children.values())
            if node_children:
                for desc in _bfs_descendants(node):
                    did = desc.node_idx
                    if did in flat_start and len(desc.input_ids) > 0:
                        q_ranges.append((flat_start[did], flat_end[did]))
                        k_ranges.append(node_range)
                        mask_types.append("full")

    # Assemble packed tensors.
    tree_packed_tokens = (
        torch.cat(flat_pieces) if flat_pieces else torch.empty(0, dtype=samples[0].dtype, device=device)
    )
    tree_packed_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    tree_packed_labels_tensor = (
        torch.cat(flat_label_pieces) if flat_label_pieces else torch.zeros_like(tree_packed_tokens)
    )
    if flat_pid_pieces is not None:
        tree_packed_position_ids = torch.cat(flat_pid_pieces)
    else:
        tree_packed_position_ids = (
            torch.cat(default_pid_pieces) if default_pid_pieces else torch.empty(0, dtype=torch.long, device=device)
        )

    # Leaf ranges + ancestor chains.
    leaf_ranges: list[RangeSpec] = []
    leaf_to_sample_list: list[int] = []
    leaf_ancestor_ranges: list[list[RangeSpec]] = []

    # Leaf ranges + ancestor chains — iterate subtrie.leaf_node_ids (DFS order)
    # so leaf_ranges align with restore's subtrie.leaf_to_sample indexing.
    leaf_ranges: list[RangeSpec] = []
    leaf_to_sample_list: list[int] = []
    leaf_ancestor_ranges: list[list[RangeSpec]] = []

    for nid in subtrie.leaf_node_ids:
        if nid not in flat_start:
            continue
        sids = leaf_node_id_to_samples.get(nid, [])
        if not sids:
            continue

        chain: list[RangeSpec] = []
        cur = next((n for n in subtrie.nodes if n.node_idx == nid), None)
        if cur is not None:
            anc = cur.ancestor
            while anc is not None and anc in subtrie.nodes:
                cid = anc.node_idx
                if cid in flat_start:
                    chain.append((flat_start[cid], flat_end[cid]))
                anc = anc.ancestor
        chain.reverse()

        rep_range: RangeSpec = (flat_start[nid], flat_end[nid])

        leaf_ranges.append(rep_range)
        leaf_to_sample_list.append(sids[0])
        leaf_ancestor_ranges.append(chain)

        zero_range: RangeSpec = (rep_range[1], rep_range[1])
        for dup_sid in sids[1:]:
            leaf_ranges.append(zero_range)
            leaf_to_sample_list.append(dup_sid)
            leaf_ancestor_ranges.append(chain + [rep_range])

    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample_list, leaf_ranges, strict=False)}
    prefix_range = (flat_start[root_nodes[0].node_idx], flat_end[root_nodes[0].node_idx])

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
        boundary_registry=prepare_packed_label(
            samples, root_nodes, valid_ids, leaf_node_id_to_samples, flat_end, owner_offset
        ),
    )
    params._leaf_ancestor_ranges = leaf_ancestor_ranges

    return params


def _bfs_descendants(node: TrieNode) -> list[TrieNode]:
    """All descendants of node in BFS order (excluding node itself)."""
    result: list[TrieNode] = []
    queue = list(node.children.values())
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(n.children.values())
    return result
