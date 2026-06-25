# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
# Copyright 2025-2026 The AReaL Authors (Ant Group, Tsinghua University, HKUST)
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

"""Dynamic-trie prefix-tree builder.

Token-by-token trie insertion supporting arbitrary tree depth. Detects the
shared-prefix tree directly from input tokens; no rollout-side metadata
required. Invoked by
:func:`verl.utils.prefix_tree.magi.build_prefix_tree_micro_batch` when
when ``prefix_segments_batch`` is not provided.

Algorithm originally derived from AReaL
(https://github.com/inclusionAI/AReaL).
"""

from __future__ import annotations

import logging as _logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_torch_device
from verl.utils.prefix_tree.utils import TreeNode
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
    roundup_divisible,
)

__all__ = [
    "build_tree_dynamic",
    "build_mini_batch_prefix_groups",
    "dfs_leaf_order",
    "dfs_micro_batch_groups",
    "compute_prefix_sharing_ratio",
    # Lower-level helpers exposed for testing / benchmarking
    "TrieNode",
    "greedy_build_tries",
    "mbs_groups_from_trie",
    "trie_group_attn_area",
    "convert_trie_to_tree_node",
    # Load balancing
    "trie_group_flat_tokens",
    "get_prefix_tree_mbs_dp_partitions",
    "get_dfs_balanced_partitions",
    "get_prefix_balanced_partitions",
    "reorder_and_balance_for_prefix_tree",
]


@dataclass
class TrieNode:
    """Compressed-trie node (after `_compress` pass).

    Each non-root node represents a contiguous run of tokens shared by the same
    set of sequences. Root has ``start_idx == end_idx == -1`` and stores no
    tokens — children are accessed via ``.children: dict[first_token, TrieNode]``.
    """

    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.start_idx == -1 and self.end_idx == -1


class _BuildNode:
    """Internal — temporary uncompressed node used during insertion."""

    __slots__ = ("tree_id", "token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> None:
    current = root
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _BuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


def _compress_trie(root: _BuildNode) -> TrieNode:
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(node: _BuildNode, ancestors: list[TrieNode]) -> TrieNode:
        tokens: list[int] = []
        current = node
        start_id = node.node_id
        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            next_child = next(iter(current.children.values()))
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError("Sequence IDs mismatch along chain")
            if next_child.node_id != current.node_id + 1:
                raise ValueError("Node IDs not consecutive along chain")
            current = next_child

        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)
        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(child, ancestors + [trie_node])
        return trie_node

    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, [])
    return trie_root


def greedy_build_tries(
    sequences: list[list[int]],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    """Token-by-token greedy trie packing across samples.

    Args:
        sequences: per-sample token lists.
        max_tokens_per_tree: upper bound on uncompressed nodes per tree (set to
            a huge value when you want a single forest).

    Returns:
        (tries, num_tokens_list) — list of compressed TrieNode roots + total
        uncompressed nodes per tree.
    """
    forests: list[dict[str, Any]] = []
    for seq_id, seq in enumerate(sequences):
        inserted = False
        for tree_id, tree in enumerate(forests):
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(tree["root"], tree["all_nodes"], seq, tree_id, seq_id)
                tree["nodes"] += additional
                inserted = True
                break
        if inserted:
            continue
        if len(seq) > max_tokens_per_tree:
            raise ValueError(f"Sequence length {len(seq)} exceeds max_tokens_per_tree {max_tokens_per_tree}")
        new_tree_id = len(forests)
        new_root = _BuildNode(new_tree_id, -1, -1)
        all_nodes: list[_BuildNode] = []
        _insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    tries = [_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]
    return tries, num_tokens_list


def convert_trie_to_tree_node(
    trie: TrieNode,
) -> Optional[tuple[TreeNode, list[int]]]:
    """Convert a compressed trie to ``(TreeNode, leaf_to_sample)`` consumed by
    :func:`verl.utils.prefix_tree.utils.build_layout_from_tree_node`.

    The trie root is a virtual placeholder with no tokens. We promote the
    trie's only child as the TreeNode root so the downstream flex-spec
    builder sees a non-zero root segment.

    Returns ``None`` when there's no real sharing (single sample, no children,
    or multi-forest case).

    Returns ``(root, leaf_to_sample)`` where ``leaf_to_sample[i]`` is the
    original sample index for the i-th leaf in DFS pre-order — matches the
    contract expected by ``build_layout_from_tree_node``.
    """
    if not trie.children:
        _logging.getLogger(__name__).warning(
            "prefix_tree: convert_trie_to_tree_node: trie has no children — no sharing, returning None"
        )
        return None
    if len(trie.children) > 1:
        _logging.getLogger(__name__).warning(
            "prefix_tree: convert_trie_to_tree_node: multiple roots (multi-forest, %d roots) — returning None",
            len(trie.children),
        )
        return None

    leaf_to_sample: list[int] = []

    def _convert(trie_node: TrieNode) -> TreeNode:
        segment_len = len(trie_node.tokens)
        children: list[TreeNode] = [_convert(child) for _tok, child in sorted(trie_node.children.items())]
        node = TreeNode(segment_len=segment_len, children=children)
        if not children:
            if len(trie_node.sequence_ids) != 1:
                # Create zero-length leaves for ALL duplicate sequences (incl. first)
                # so leaf_to_sample count matches leaf count in TreeNode DFS walk
                for sid in trie_node.sequence_ids:
                    leaf_to_sample.append(sid)
                    node.children.append(TreeNode(segment_len=0, children=[]))
            else:
                leaf_to_sample.append(trie_node.sequence_ids[0])
        else:
            # Samples that terminate at this intermediate node need zero-length leaves.
            child_ids = {sid for c in trie_node.children.values() for sid in c.sequence_ids}
            for sid in trie_node.sequence_ids:
                if sid not in child_ids:
                    leaf_to_sample.append(sid)
                    # zero-length leaf: sample's tokens are entirely in ancestors
                    node.children.append(TreeNode(segment_len=0, children=[]))
        return node

    only_child = next(iter(trie.children.values()))
    try:
        root = _convert(only_child)
    except ValueError:
        return None  # duplicate sequences — fall back to standard attention
    if not root.children:
        return None
    return root, leaf_to_sample


def build_tree_dynamic(samples: list[Tensor]) -> Optional[tuple[TreeNode, list[int]]]:
    """Token-by-token trie detection. Returns ``(TreeNode, leaf_to_sample)`` or None.

    Builds a compressed trie of the input samples, then converts it to the
    canonical ``TreeNode`` representation consumed by
    :func:`verl.utils.prefix_tree.utils.build_layout_from_tree_node`.

    ``leaf_to_sample[i]`` gives the original sample index for the i-th leaf in
    DFS pre-order. Returns ``None`` when there's no shared prefix (empty input,
    single sample, or multi-forest case).
    """
    if not samples:
        return None
    sequences = [t.tolist() for t in samples]
    max_tokens_per_tree = sum(len(s) for s in sequences) * 10  # one forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens_per_tree)
    if not tries or len(tries) > 1:
        _logging.getLogger(__name__).warning(
            "prefix_tree: build_tree_dynamic: multi-forest or empty tries (len=%d) — no shared prefix, returning None",
            len(tries),
        )
        return None
    return convert_trie_to_tree_node(tries[0])


def _trie_seq_ids(node: TrieNode) -> list[int]:
    """Collect all sequence IDs from leaf nodes of a compressed-trie subtree."""
    if not node.children:
        return list(node.sequence_ids)
    ids: list[int] = []
    for child in node.children.values():
        ids.extend(_trie_seq_ids(child))
    return ids


def _trie_effective_tokens(node: TrieNode) -> int:
    """Total unique (flat) tokens in a compressed-trie subtree.

    Counts the tokens owned by this node plus all its descendants — i.e. the
    flat token count that would result from laying this subtree out for
    prefix-tree attention.
    """
    total = len(node.tokens)
    for child in node.children.values():
        total += _trie_effective_tokens(child)
    return total


def trie_group_flat_tokens(group: list[int], trie: TrieNode) -> int:
    """Flat (deduplicated) token count for a subset of sequences within a trie.

    Counts tokens on the minimal sub-trie spanning exactly the sequences in
    ``group`` — i.e. the effective forward-pass token budget when those
    sequences are processed together with prefix sharing.

    Args:
        group: Sequence indices as stored in ``TrieNode.sequence_ids``.
        trie: Root of the compressed trie (``trie.is_root == True``).

    Returns:
        Total number of unique tokens required to process this group.
    """
    keep = frozenset(group)

    def _count(node: TrieNode) -> int:
        if not node.children:
            return len(node.tokens) if any(s in keep for s in node.sequence_ids) else 0
        has_relevant = False
        relevant_total = 0
        for child in node.children.values():
            if any(s in keep for s in child.sequence_ids):
                has_relevant = True
                relevant_total += _count(child)
        return relevant_total + len(node.tokens) if has_relevant else 0

    return sum(_count(c) for c in trie.children.values())


def _leaf_attn_area_inc(node: TrieNode, covered: set[int]) -> tuple[int, list[TrieNode]]:
    """Incremental non-masked attention area for adding *node* (a leaf) to a group.

    Derived from the prefix-tree block mask (same three block types as
    ``_build_flex_key``):
        - new ancestor self-attention  (paid once when first seen)
        - leaf attending to all its ancestors
        - leaf self-attention

        inc = new_anc_len²  +  (shared_anc_len + new_anc_len) × leaf_len  +  leaf_len²

    Works correctly for single-level and multi-level trees, and for groups that
    mix multiple prefix families (different subtrees of the same trie root).

    Returns:
        (inc, new_anc): incremental area cost and the newly-covered ancestor nodes.
    """
    new_anc = [n for n in node.ancestors if id(n) not in covered]
    shared_anc_len = sum(len(n.tokens) for n in node.ancestors if id(n) in covered)
    new_anc_len = sum(len(n.tokens) for n in new_anc)
    leaf_len = len(node.tokens)
    inc = new_anc_len * new_anc_len + (shared_anc_len + new_anc_len) * leaf_len + leaf_len * leaf_len
    return inc, new_anc


def trie_group_attn_area(group: list[int], trie: TrieNode) -> int:
    """Total non-masked attention area for a subset of sequences within a trie.

    Mirrors :func:`trie_group_flat_tokens` but returns the attention area instead
    of flat token count.  The area equals ``sqrt(area)`` effective tokens — the
    budget unit used by :func:`mbs_groups_from_trie` when ``use_n2_cost=True``.
    """
    keep = frozenset(group)
    covered: set[int] = set()
    total = 0

    def _visit(node: TrieNode) -> None:
        nonlocal total
        if not node.children:
            if any(s in keep for s in node.sequence_ids):
                inc, new_anc = _leaf_attn_area_inc(node, covered)
                total += inc
                covered.update(id(n) for n in new_anc + [node])
        else:
            for child in node.children.values():
                if any(s in keep for s in child.sequence_ids):
                    _visit(child)

    for child in trie.children.values():
        if any(s in keep for s in child.sequence_ids):
            _visit(child)
    return total


def build_mini_batch_prefix_groups(
    sequences: list[list[int]],
) -> list[tuple[list[int], int]]:
    """Build a mini-batch trie and return prefix-sharing groups for load balancing.

    Builds a single compressed trie over all sequences, then walks down the
    shared spine to the first branching point.  Each child subtree at that
    branching point becomes one group: the sequences in the group all share a
    common prefix, so they should be assigned to the *same* DP rank / micro-batch
    to benefit from prefix deduplication.

    Args:
        sequences: per-sample token lists (the full mini-batch).

    Returns:
        List of ``(seq_indices, effective_token_count)`` pairs.

        * ``seq_indices`` — original sample indices that share a common prefix.
        * ``effective_token_count`` — flat (deduplicated) token count for this
          group: shared prefix length + unique branch tokens.  Use this instead
          of raw sequence lengths for load-balancing workload estimates.
    """
    if not sequences:
        return []

    max_tokens = sum(len(s) for s in sequences) * 10  # one big forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens)

    groups: list[tuple[list[int], int]] = []
    for trie_root in tries:
        # Walk the spine (single-child path) accumulating the shared prefix length.
        # trie_root is a virtual root (is_root=True, no tokens).
        shared_len = 0
        current = trie_root
        while len(current.children) == 1:
            child = next(iter(current.children.values()))
            shared_len += len(child.tokens)
            current = child

        if not current.children:
            # Leaf — all sequences in this trie share the full path (e.g. identical
            # sequences, or a single sequence).
            groups.append((list(current.sequence_ids), shared_len))
        else:
            # Branching node — each child subtree is an independent group.
            for child in current.children.values():
                seq_ids = _trie_seq_ids(child)
                effective = shared_len + _trie_effective_tokens(child)
                groups.append((seq_ids, effective))

    return groups


def dfs_leaf_order(
    sequences: list[list[int]],
) -> list[int]:
    """Return sample indices in DFS pre-order of the mini-batch trie.

    Builds one trie over all sequences then walks leaves in DFS pre-order
    (children sorted by first token).  Consecutive indices in the returned
    list share the longest possible common prefix — the ideal ordering for
    prefix-tree load balancing and micro-batch grouping.

    Args:
        sequences: per-sample token lists.

    Returns:
        List of sample indices in DFS pre-order (length == len(sequences)).
    """
    if not sequences:
        return []

    max_tokens = sum(len(s) for s in sequences) * 10
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens)

    ordered: list[int] = []

    def _walk(node: TrieNode) -> None:
        if not node.children:
            ordered.extend(node.sequence_ids)
        else:
            for child in node.children.values():
                _walk(child)

    for trie_root in tries:
        for child in trie_root.children.values():
            _walk(child)

    return ordered


def dfs_micro_batch_groups(
    sequences: list[list[int]],
    max_token_len: int,
) -> list[list[int]]:
    """Group sequences into micro-batches in DFS trie order, budgeted by flat trie tokens.

    Builds ONE trie over all sequences, then traverses leaves in DFS pre-order.
    Greedily packs leaves into micro-batches: the **budget is flat (deduplicated)
    trie tokens** — prefix counted once + unique branch tokens — not raw sequence
    lengths.  This means a micro-batch of k sequences that share a long common
    prefix uses far fewer budget tokens than k × seq_len, allowing more sequences
    per batch.

    Algorithm:
        - Maintain ``covered``: set of trie-node IDs already in the current batch.
        - For each leaf in DFS pre-order, its incremental cost =
          tokens in (leaf.ancestors + [leaf]) that are NOT yet in ``covered``.
        - If incremental cost fits within remaining budget: add to current batch.
        - Else: flush current batch, start a new one with this leaf.

    Args:
        sequences: per-sample token lists (the full mini-batch).
        max_token_len: flat-token budget per micro-batch.

    Returns:
        List of micro-batch groups; each group is a list of sample indices in
        DFS pre-order.
    """
    if not sequences:
        return []

    max_tokens = sum(len(s) for s in sequences) * 10  # one big forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens)

    all_groups: list[list[int]] = []

    for trie_root in tries:
        current_group: list[int] = []
        covered: set[int] = set()  # id(TrieNode) already counted in current batch
        current_eff = 0

        def _visit(node: TrieNode) -> None:
            nonlocal current_eff
            if not node.children:
                # Leaf: full path = ancestors (list of TrieNodes) + self
                path = node.ancestors + [node]
                new_nodes = [n for n in path if id(n) not in covered]  # noqa: B023
                inc = sum(len(n.tokens) for n in new_nodes)

                if current_group and current_eff + inc > max_token_len:  # noqa: B023
                    # Flush — start a fresh batch with this leaf
                    all_groups.append(current_group[:])  # noqa: B023
                    current_group.clear()  # noqa: B023
                    covered.clear()  # noqa: B023
                    current_eff = 0
                    new_nodes = path  # all nodes are new in empty batch
                    inc = sum(len(n.tokens) for n in new_nodes)

                current_group.extend(node.sequence_ids)  # noqa: B023
                covered.update(id(n) for n in new_nodes)  # noqa: B023
                current_eff += inc
            else:
                for child in node.children.values():  # sorted by token key
                    _visit(child)

        for child in trie_root.children.values():
            _visit(child)

        if current_group:
            all_groups.append(current_group)

    return all_groups


def trie_dfs_leaf_order(trie: TrieNode) -> list[int]:
    """Return sample indices in DFS pre-order from an existing trie."""
    ordered: list[int] = []

    def _walk(node: TrieNode) -> None:
        if not node.children:
            ordered.extend(node.sequence_ids)
        else:
            for child in node.children.values():
                _walk(child)

    if trie.is_root:
        for child in trie.children.values():
            _walk(child)
    else:
        _walk(trie)
    return ordered


def trie_to_leaf_ids(trie: TrieNode) -> np.ndarray:  # noqa: F821
    """Return np.array mapping sample_idx → leaf_id (DFS order)."""
    leaf_id = [-1]  # counter
    mapping: dict[int, int] = {}

    def _walk(node: TrieNode) -> None:
        if not node.children:
            for sid in node.sequence_ids:
                mapping[sid] = leaf_id[0]
            leaf_id[0] += 1
        else:
            for child in node.children.values():
                _walk(child)

    if trie.is_root:
        for child in trie.children.values():
            _walk(child)
    else:
        _walk(trie)

    max_idx = max(mapping.keys()) if mapping else -1
    result = np.full(max_idx + 1, -1, dtype=np.int32)
    for sid, lid in mapping.items():
        result[sid] = lid
    return result


def mbs_groups_from_trie(
    trie: TrieNode,
    max_token_len: int,
    use_n2_cost: bool = False,
) -> list[list[int]]:
    """Group sequences into micro-batches from an existing trie.

    When ``use_n2_cost=False`` (default): budget is flat (deduplicated) tokens.

    When ``use_n2_cost=True``: budget is the *effective attention length*
    ``sqrt(non_masked_area)`` where the non-masked area matches the prefix-tree
    block mask exactly — prefix self-attention + each leaf attending to all its
    ancestors + leaf self-attention.  For each new leaf the incremental area is:

        inc = new_anc_len²  +  (shared_anc_len + new_anc_len) × leaf_len  +  leaf_len²

    where ``new_anc_len`` are ancestor tokens not yet in ``covered`` (paid once,
    like the mask's prefix block) and ``shared_anc_len`` are already covered.
    Budget check: ``total_area + inc > max_token_len²`` (avoids sqrt in hot path).
    This handles arbitrary tree depth and multi-family groups correctly.

    When a trie leaf holds multiple sequence_ids (identical sequences), all IDs
    stay in the same DFS group.  prune_trie returns None for that group
    (len(kept)>1), causing FA3 fallback for that one group only.  Keeping
    duplicates in-group avoids an extra singleton group that would cause
    same_micro_num_in_dp to pad the other DP rank, double-counting its gradients.
    """
    all_groups: list[list[int]] = []
    budget_sq = max_token_len * max_token_len  # used only when use_n2_cost=True

    for trie_root in [trie]:
        current_group: list[int] = []
        covered: set[int] = set()
        current_eff = 0  # flat tokens (use_n2_cost=False) or accumulated area (True)

        def _visit(node: TrieNode) -> None:
            nonlocal current_eff
            if not node.children:
                if use_n2_cost:  # noqa: B023
                    inc, new_anc = _leaf_attn_area_inc(node, covered)  # noqa: B023
                    if current_group and current_eff + inc > budget_sq:  # noqa: B023
                        all_groups.append(current_group[:])  # noqa: B023
                        current_group.clear()  # noqa: B023
                        covered.clear()  # noqa: B023
                        current_eff = 0
                        inc, new_anc = _leaf_attn_area_inc(node, covered)  # recompute after flush  # noqa: B023
                    current_group.extend(node.sequence_ids)  # noqa: B023
                    covered.update(id(n) for n in new_anc + [node])  # noqa: B023
                    current_eff += inc
                else:
                    path = node.ancestors + [node]
                    new_nodes = [n for n in path if id(n) not in covered]  # noqa: B023
                    inc = sum(len(n.tokens) for n in new_nodes)
                    if current_group and current_eff + inc > max_token_len:  # noqa: B023
                        all_groups.append(current_group[:])  # noqa: B023
                        current_group.clear()  # noqa: B023
                        covered.clear()  # noqa: B023
                        current_eff = 0
                        new_nodes = path
                        inc = sum(len(n.tokens) for n in new_nodes)
                    # Keep all sequence_ids (including duplicates) in the same group.
                    # Duplicates land in the same group as their originals → prune_trie
                    # returns None for that group → FA3 fallback for that one group.
                    # This avoids creating an extra singleton group that would upset
                    # same_micro_num_in_dp and double-count gradients on the other rank.
                    current_group.extend(node.sequence_ids)  # noqa: B023
                    covered.update(id(n) for n in new_nodes)  # noqa: B023
                    current_eff += inc
            else:
                for child in node.children.values():
                    _visit(child)

        if trie_root.is_root:
            for child in trie_root.children.values():
                _visit(child)
        else:
            _visit(trie_root)

        if current_group:
            all_groups.append(current_group)

    return all_groups


def prune_trie(
    trie: TrieNode,
    keep_leaf_ids: set[int],
) -> Optional[tuple[TreeNode, list[int]]]:  # noqa: F821
    """Extract a subtree containing only the given leaf sample indices.

    Returns ``(tree_root, leaf_to_sample)`` ready for
    :func:`build_layout_from_tree_node`, or ``None`` if no leaves match.
    """
    if not keep_leaf_ids:
        return None

    def _build_subtree(node: TrieNode) -> Optional[TreeNode]:
        segment_len = len(node.tokens)
        children: list[TreeNode] = []

        if not node.children:
            # Leaf: keep only the sequence_ids that are in keep_leaf_ids.
            kept = [s for s in node.sequence_ids if s in keep_leaf_ids]
            if not kept:
                return None
            if len(kept) > 1:
                # Two identical sequences in this micro-batch share the same trie
                # leaf — can't represent as a single TreeNode.  Fall back.
                return None
            leaf_to_sample.extend(kept)
            return TreeNode(segment_len=segment_len, children=[])

        # Internal node: keep children that intersect keep_leaf_ids
        for child in node.children.values():
            if keep_leaf_ids.isdisjoint(child.sequence_ids):
                continue
            child_tree = _build_subtree(child)
            if child_tree is not None:
                children.append(child_tree)

        if segment_len == 0 and len(children) <= 1:
            # zero-length internal node with ≤1 child → skip (chain compression)
            if len(children) == 1:
                return children[0]
            return None

        return TreeNode(segment_len=segment_len, children=children)

    leaf_to_sample: list[int] = []
    root = _build_subtree(trie)
    if root is None or not root.children and not root.is_leaf:
        return None
    # Guard: if any keep_leaf_id was dropped (e.g. duplicate sequences sharing a
    # trie leaf), fall back so restore_flat_to_nested never sees missing slots.
    if set(leaf_to_sample) != keep_leaf_ids:
        _logging.getLogger(__name__).warning(
            "prefix_tree: prune_trie: duplicate-leaf mismatch (got %s, expected %s) — returning None",
            set(leaf_to_sample),
            keep_leaf_ids,
        )
        return None
    return root, leaf_to_sample


def compute_prefix_sharing_ratio(input_ids, attention_mask=None) -> float:
    """Compute prefix-sharing ratio from a batch of token sequences.

    ratio = 1 - flat_trie_tokens / total_raw_tokens

    ``flat_trie_tokens`` counts each shared token exactly once (the trie
    representation); ``total_raw_tokens`` is the sum of all sequence lengths.
    A ratio of 0 means no sharing; 1 means every token is shared.

    Args:
        input_ids: One of:
            - ``torch.nested`` ``NestedTensor`` (jagged, SFT no-padding path)
            - ``list[list[int]]`` of raw token sequences
            - 2-D padded ``Tensor`` of shape ``(batch, max_len)`` — requires
              ``attention_mask`` to trim padding
        attention_mask: Optional 2-D ``Tensor`` ``(batch, max_len)`` used to
            trim padding when ``input_ids`` is a padded 2-D tensor.

    Returns:
        ``float`` in ``[0, 1]``.  Returns ``0.0`` for empty or single-sequence
        input, or when no prefix is shared.
    """
    if isinstance(input_ids, Tensor) and input_ids.is_nested:
        sequences = [t.tolist() for t in input_ids.unbind()]
    elif isinstance(input_ids, Tensor) and input_ids.dim() == 2:
        # Padded tensor path (PPO rollout output)
        seqlens = (
            attention_mask.sum(dim=-1).tolist()
            if attention_mask is not None
            else [input_ids.shape[1]] * input_ids.shape[0]
        )
        sequences = [input_ids[i, : int(seqlens[i])].tolist() for i in range(input_ids.shape[0])]
    elif isinstance(input_ids, list):
        sequences = input_ids
    else:
        return 0.0

    total_raw = sum(len(s) for s in sequences)
    if total_raw == 0:
        return 0.0

    _, num_tokens = greedy_build_tries(sequences, max_tokens_per_tree=total_raw * 10)
    return 1.0 - sum(num_tokens) / total_raw


def compute_prefix_tree_metrics(
    input_ids,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    dynbsz_estimator: str = "length",
    micro_batch_size: int = 0,
) -> dict:
    """Compute prefix-tree metrics as a ``prefix_tree/`` namespace dict.

    Returns a dict with keys:
        ``prefix_tree/global_shared_ratio``      — fraction of tokens saved by deduplication
        ``prefix_tree/micro_batch_shared_ratio`` — mean per-micro-batch sharing ratio;
                                                   computed from trie groups (dynbsz) or
                                                   consecutive slices of ``micro_batch_size`` (fixed mbs)
        ``prefix_tree/flat_tokens``              — deduplicated flat trie token count
        ``prefix_tree/raw_tokens``               — total raw token count across all sequences
        ``prefix_tree/avg_mbs``                  — avg sequences per micro-batch (dynbsz only)

    Args:
        input_ids: NestedTensor, padded 2-D Tensor, or list[list[int]].
        attention_mask: Optional mask for padded 2-D case.
        max_token_len_per_gpu: dynbsz budget; if set, computes trie-based micro-batch groups.
        dynbsz_estimator: ``"length"`` (flat tokens) or ``"area"`` (sqrt of attention area).
        micro_batch_size: fixed micro-batch size (sequences per micro-batch) for non-dynbsz mode.

    Returns:
        dict of float metrics, all zero if no sequences.
    """
    if isinstance(input_ids, Tensor) and input_ids.is_nested:
        sequences = [t.tolist() for t in input_ids.unbind()]
    elif isinstance(input_ids, Tensor) and input_ids.dim() == 2:
        seqlens = (
            attention_mask.sum(dim=-1).tolist()
            if attention_mask is not None
            else [input_ids.shape[1]] * input_ids.shape[0]
        )
        sequences = [input_ids[i, : int(seqlens[i])].tolist() for i in range(input_ids.shape[0])]
    elif isinstance(input_ids, list):
        sequences = input_ids
    else:
        return {
            "prefix_tree/global_shared_ratio": 0.0,
            "prefix_tree/flat_tokens": 0,
            "prefix_tree/raw_tokens": 0,
        }

    total_raw = sum(len(s) for s in sequences)
    if total_raw == 0:
        return {
            "prefix_tree/global_shared_ratio": 0.0,
            "prefix_tree/flat_tokens": 0,
            "prefix_tree/raw_tokens": 0,
        }

    _, num_tokens = greedy_build_tries(sequences, max_tokens_per_tree=total_raw * 10)
    flat = sum(num_tokens)
    result = {
        "prefix_tree/global_shared_ratio": 1.0 - flat / total_raw,
        "prefix_tree/flat_tokens": flat,
        "prefix_tree/raw_tokens": total_raw,
    }

    def _micro_batch_ratio(groups_iter) -> float | None:
        ratios = []
        for group_seqs in groups_iter:
            group_raw = sum(len(s) for s in group_seqs)
            if group_raw == 0:
                continue
            _, gnum = greedy_build_tries(group_seqs, max_tokens_per_tree=group_raw * 10)
            ratios.append(1.0 - sum(gnum) / group_raw)
        return sum(ratios) / len(ratios) if ratios else None

    if max_token_len_per_gpu is not None:
        tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=total_raw * 10)
        if tries:
            trie = tries[0]
            use_n2 = dynbsz_estimator == "area"
            groups = mbs_groups_from_trie(trie, max_token_len_per_gpu, use_n2_cost=use_n2)
            result["prefix_tree/avg_mbs"] = len(sequences) / len(groups) if groups else 0.0
            # Per-micro-batch sharing ratio using actual dynbsz groups and the global trie.
            ratios = []
            for group in groups:
                group_raw = sum(len(sequences[i]) for i in group)
                if group_raw == 0:
                    continue
                ratios.append(1.0 - trie_group_flat_tokens(group, trie) / group_raw)
            if ratios:
                result["prefix_tree/micro_batch_shared_ratio"] = sum(ratios) / len(ratios)
    elif micro_batch_size > 0 and len(sequences) >= micro_batch_size:
        # Fixed micro-batch size: consecutive slices.
        groups_iter = (sequences[i : i + micro_batch_size] for i in range(0, len(sequences), micro_batch_size))
        ratio = _micro_batch_ratio(groups_iter)
        if ratio is not None:
            result["prefix_tree/micro_batch_shared_ratio"] = ratio

    return result


def prepare_prefix_tree_micro_batches(
    data,
    sp_size: int,
    dp_group=None,
    same_micro_num_in_dp: bool = True,
    num_batches_divided_by: int | None = None,
):
    """Prepare micro-batches using prefix-tree DFS grouping.

    Expects a pre-built trie stored via ``tu.assign_non_tensor(data, prefix_tree=trie)``.
    If not present, builds one from sequences (backward-compatible fallback).
    """
    _logging.getLogger(__name__).warning_once(
        "prefix_tree is on: max_token_len_per_gpu is interpreted as "
        "deduplicated (flat trie) token count (length estimator) or "
        "sqrt(non_masked_attention_area) (area estimator), not raw sequence length."
    )

    assert "max_token_len_per_gpu" in data.keys(), "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
    max_token_len_per_gpu = data["max_token_len_per_gpu"]

    estimator = tu.get_non_tensor_data(data, "prefix_tree_dynbsz_length_estimator", default="length")
    use_n2_cost = estimator == "area"

    # After pre-dispatch each CP rank processes flat_tokens/CP tokens, so the total
    # budget across all CP ranks is max_token_len_per_gpu × sp_size for both estimators.
    max_token_len = max_token_len_per_gpu * sp_size

    trie = tu.get_non_tensor_data(data, "prefix_tree", default=None)
    if trie is not None:
        batch_idx_list = mbs_groups_from_trie(trie, max_token_len, use_n2_cost=use_n2_cost)
    else:
        input_ids = data["input_ids"]
        seqs = [t.tolist() for t in input_ids.unbind()]
        batch_idx_list = dfs_micro_batch_groups(seqs, max_token_len)

    if torch.distributed.is_initialized() and same_micro_num_in_dp and dp_group is not None:
        n_mb = torch.tensor([len(batch_idx_list)], device=get_torch_device().current_device())
        torch.distributed.all_reduce(n_mb, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        while len(batch_idx_list) < n_mb.item():
            batch_idx_list.append(batch_idx_list[-1])

    if num_batches_divided_by is not None:
        target = roundup_divisible(len(batch_idx_list), num_batches_divided_by)
        while len(batch_idx_list) < target:
            batch_idx_list.append(batch_idx_list[-1])

    micro_batches = [tu.index_select_tensor_dict(data, idx) for idx in batch_idx_list]
    # Reorder micro-batches in inc-then-dec flat-token pattern to reduce PP bubble.
    # Preserves prefix locality: samples within a group share prefixes and stay together.
    if trie is not None and len(batch_idx_list) > 1:

        def _group_flat_tokens(group: list[int]) -> int:
            keep = set(group)

            def _count(node):
                total = 0
                if not node.children:
                    return len(node.tokens) if any(s in keep for s in node.sequence_ids) else 0
                kept = False
                for child in node.children.values():
                    if any(s in keep for s in child.sequence_ids):
                        kept = True
                        total += _count(child)
                return total + len(node.tokens) if kept else total

            return sum(_count(c) for c in trie.children.values())

        tokens_per_group = [_group_flat_tokens(g) for g in batch_idx_list]
        sorted_groups = sorted(zip(tokens_per_group, batch_idx_list, range(len(batch_idx_list)), strict=False))
        ordered = [g for _, g, _ in sorted_groups]
        batch_idx_list = ordered[::2] + ordered[1::2][::-1]
        micro_batches = [tu.index_select_tensor_dict(data, idx) for idx in batch_idx_list]
    # Attach pruned subtree to each micro-batch for downstream trie reuse.
    if trie is not None:
        for idx, mb in zip(batch_idx_list, micro_batches, strict=False):
            subtree = prune_trie(trie, set(idx))
            if subtree is not None:
                tree_root, leaf_to_sample_global = subtree
                global_to_local = {g: loc for loc, g in enumerate(idx)}
                leaf_to_sample_local = [global_to_local[g] for g in leaf_to_sample_global]
                tu.assign_non_tensor(mb, prefix_tree_subtree=(tree_root, leaf_to_sample_local))
    return micro_batches, batch_idx_list


def _is_prefix_tree_enabled(config_or_data) -> bool:
    if isinstance(config_or_data, dict):
        return config_or_data.get("use_prefix_tree", False)
    return getattr(config_or_data, "use_prefix_tree", False)


def get_dfs_balanced_partitions(
    data,
    config_or_data: dict,
    dp_size: int,
    *,
    attention_mask=None,
    contiguous_partitions: bool = False,
):
    """Re-order batch in DFS trie order and return balanced partitions."""
    if not _is_prefix_tree_enabled(config_or_data):
        return None

    batch_size = data.batch["input_ids"].shape[0] if hasattr(data, "batch") else len(data["input_ids"])
    _ids = data.batch["input_ids"] if hasattr(data, "batch") else data["input_ids"]
    _mask = (
        attention_mask
        if attention_mask is not None
        else (data.batch.get("attention_mask", None) if hasattr(data, "batch") else None)
    )

    if _mask is not None:
        seqs = [_ids[i][_mask[i].bool()].tolist() or [0] for i in range(batch_size)]
    else:
        seqs = [_ids[i].tolist() for i in range(batch_size)]

    dfs_order = dfs_leaf_order(seqs)
    if len(dfs_order) < batch_size:
        missing = [i for i in range(batch_size) if i not in set(dfs_order)]
        dfs_order = dfs_order + missing

    if hasattr(data, "reorder"):
        data.reorder(torch.tensor(dfs_order))
    else:
        data = tu.index_select_tensor_dict(data, torch.tensor(dfs_order))

    if hasattr(data, "batch") and "attention_mask" in data.batch:
        global_seqlen_lst = data.batch["attention_mask"].view(batch_size, -1).sum(-1)
    else:
        global_seqlen_lst = torch.Tensor([item.size()[0] for item in data["input_ids"]])

    if contiguous_partitions:
        per_rank = batch_size // dp_size
        partition_lst = [list(range(i * per_rank, (i + 1) * per_rank)) for i in range(dp_size)]
    else:
        partition_lst = get_seqlen_balanced_partitions(
            calculate_workload(global_seqlen_lst), k_partitions=dp_size, equal_size=True
        )

    return partition_lst, global_seqlen_lst, data


def get_prefix_balanced_partitions(
    sequences: list[list[int]],
    k_partitions: int,
) -> list[list[int]]:
    """Partition sequences into k groups using mini-batch trie grouping."""
    if not sequences:
        return [[] for _ in range(k_partitions)]

    groups = build_mini_batch_prefix_groups(sequences)

    group_workloads = [int(calculate_workload(torch.tensor([eff])).item()) for _, eff in groups]

    if len(groups) <= k_partitions:
        partitions: list[list[int]] = [list(seq_ids) for seq_ids, _ in groups]
        while len(partitions) < k_partitions:
            partitions.append([])
        return partitions

    group_partitions = get_seqlen_balanced_partitions(
        seqlen_list=group_workloads,
        k_partitions=k_partitions,
        equal_size=False,
    )

    sample_partitions = []
    for gp in group_partitions:
        sample_indices: list[int] = []
        for gi in gp:
            sample_indices.extend(groups[gi][0])
        sample_partitions.append(sorted(sample_indices))

    return sample_partitions


def get_prefix_tree_mbs_dp_partitions(
    sequences: list[list[int]],
    max_token_len: int,
    dp_size: int,
) -> tuple[list[list[list[int]]], list[int]]:
    """Partition micro-batches across DP ranks using flat-token workload balance.

    Forms micro-batches globally from the full batch trie, estimates the flat
    (deduplicated) token cost of each mbs via :func:`trie_group_flat_tokens`,
    then assigns whole mbs to DP ranks with equal mbs count per rank.

    Treating mbs as atomic units preserves the prefix-sharing ratio: no mbs is
    ever split across ranks, so the deduplicated token count computed at mbs
    formation time remains valid after DP assignment.

    Args:
        sequences: Per-sample token lists for the full global batch.
        max_token_len: Flat-token budget per micro-batch
            (``max_token_len_per_gpu * sp_size``).
        dp_size: Number of data-parallel ranks.

    Returns:
        mbs_per_rank: ``mbs_per_rank[i]`` is the list of mbs groups (each a
            ``list[int]`` of sequence indices) assigned to rank ``i``.
        new_seq_order: Flat list of original sequence indices in the order they
            should appear after ``data.reorder()`` — rank 0's sequences first,
            rank 1's next, etc.
    """

    def _fallback_equal_split():
        per_rank = len(sequences) // dp_size
        mbs_per_rank = [[[i] for i in range(r * per_rank, (r + 1) * per_rank)] for r in range(dp_size)]
        return mbs_per_rank, list(range(len(sequences)))

    max_total = sum(len(s) for s in sequences) * 10
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_total)
    if not tries:
        return _fallback_equal_split()

    trie = tries[0]
    mbs_groups = mbs_groups_from_trie(trie, max_token_len)
    if len(mbs_groups) < dp_size:
        return _fallback_equal_split()

    # Estimate flat token cost per mbs.
    mbs_flat_tokens = [trie_group_flat_tokens(g, trie) for g in mbs_groups]

    # Pad to a multiple of dp_size so equal_size=True is satisfied.
    while len(mbs_groups) % dp_size:
        mbs_groups.append(mbs_groups[-1])
        mbs_flat_tokens.append(mbs_flat_tokens[-1])

    # Assign equal mbs count per rank, balanced by flat token workload.
    mbs_assignment = get_seqlen_balanced_partitions(mbs_flat_tokens, dp_size, equal_size=True)

    mbs_per_rank: list[list[list[int]]] = []
    new_seq_order: list[int] = []
    for rank_mbs_indices in mbs_assignment:
        rank_mbs = [mbs_groups[i] for i in rank_mbs_indices]
        mbs_per_rank.append(rank_mbs)
        for grp in rank_mbs:
            new_seq_order.extend(grp)

    return mbs_per_rank, new_seq_order


def reorder_and_balance_for_prefix_tree(
    data,
    config_or_data: dict,
    dp_size: int,
    *,
    attention_mask=None,
    metrics: dict | None = None,
    logging_prefix: str = "global_seqlen",
) -> bool:
    """DFS-reorder batch and compute contiguous partitions for prefix-tree."""
    if not _is_prefix_tree_enabled(config_or_data):
        return False

    result = get_dfs_balanced_partitions(
        data,
        config_or_data,
        dp_size,
        attention_mask=attention_mask,
        contiguous_partitions=True,
    )
    if result is None:
        return False

    global_partition_lst, global_seqlen_lst, _ = result
    global_idx = torch.arange(global_seqlen_lst.shape[0])
    data.reorder(global_idx)
    if metrics is not None:
        stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(),
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(stats)
    return True
