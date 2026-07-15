# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import Tensor

from verl.utils.prefix_tree.utils import TreeNode

__all__ = [
    "build_tree_dynamic",
    "build_mini_batch_prefix_groups",
    "dfs_leaf_order",
    "dfs_micro_batch_groups",
    "compute_prefix_sharing_ratio",
    # Lower-level helpers exposed for testing / benchmarking
    "TrieNode",
    "greedy_build_tries",
    "convert_trie_to_tree_node",
    # Load balancing
    "get_dfs_balanced_partitions",
    "get_prefix_balanced_partitions",
    "reorder_and_balance_for_prefix_tree",
]


# ============================================================================
# Trie construction (token-by-token insertion)
# ============================================================================


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


# ============================================================================
# Trie → TreeNode conversion (arbitrary depth preserved)
# ============================================================================


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
        return None
    if len(trie.children) > 1:
        return None

    leaf_to_sample: list[int] = []

    def _convert(trie_node: TrieNode) -> TreeNode:
        segment_len = len(trie_node.tokens)
        children: list[TreeNode] = [_convert(child) for _tok, child in sorted(trie_node.children.items())]
        node = TreeNode(segment_len=segment_len, children=children)
        if not children:
            if len(trie_node.sequence_ids) != 1:
                # Identical sequences share a leaf (common in GRPO with n>1 rollouts).
                # Fall back to standard computation by signalling failure.
                raise ValueError(
                    f"Trie leaf has duplicate sequences {trie_node.sequence_ids}; falling back to standard attention."
                )
            leaf_to_sample.append(trie_node.sequence_ids[0])
        return node

    only_child = next(iter(trie.children.values()))
    try:
        root = _convert(only_child)
    except ValueError:
        return None  # duplicate sequences — fall back to standard attention
    if not root.children:
        return None
    return root, leaf_to_sample


# ============================================================================
# Detection entry: build_tree_dynamic
# ============================================================================


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
        return None
    return convert_trie_to_tree_node(tries[0])


# ============================================================================
# Mini-batch level prefix grouping (for DP load balancing)
# ============================================================================


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


def trie_to_leaf_ids(trie: TrieNode) -> np.ndarray:  # type: ignore[type-arg]  # noqa: F821
    """Return np.array mapping sample_idx → leaf_id (DFS order)."""
    import numpy as np

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
) -> list[list[int]]:
    """Group sequences into micro-batches from an existing trie.

    Same algorithm as :func:`dfs_micro_batch_groups` but accepts a pre-built
    trie instead of rebuilding from sequences.
    """
    all_groups: list[list[int]] = []
    for trie_root in [trie]:
        current_group: list[int] = []
        covered: set[int] = set()
        current_eff = 0

        def _visit(node: TrieNode) -> None:
            nonlocal current_eff
            if not node.children:
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
    from verl.utils.prefix_tree.utils import TreeNode

    if not keep_leaf_ids:
        return None

    def _build_subtree(node: TrieNode) -> Optional[TreeNode]:
        segment_len = len(node.tokens)
        children: list[TreeNode] = []

        if not node.children:
            # Leaf: keep if any of this leaf's sequence_ids are in keep_leaf_ids
            if not set(node.sequence_ids) & keep_leaf_ids:
                return None
            leaf_to_sample.extend(node.sequence_ids)
            return TreeNode(segment_len=segment_len, children=[])

        # Internal node: keep children that intersect keep_leaf_ids
        for child in node.children.values():
            if not set(child.sequence_ids) & keep_leaf_ids:
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
    from torch import Tensor

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


def compute_prefix_tree_metrics(input_ids, attention_mask=None) -> dict:
    """Compute prefix-tree metrics as a ``prefix_tree/`` namespace dict.

    Returns a dict with keys:
        ``prefix_tree/sharing_ratio``  — fraction of tokens saved by deduplication
        ``prefix_tree/flat_tokens``    — deduplicated flat trie token count
        ``prefix_tree/raw_tokens``     — total raw token count across all sequences

    Args:
        input_ids: NestedTensor, padded 2-D Tensor, or list[list[int]].
        attention_mask: Optional mask for padded 2-D case.

    Returns:
        dict of float metrics, all zero if no sequences.
    """
    from torch import Tensor

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
        return {"prefix_tree/sharing_ratio": 0.0, "prefix_tree/flat_tokens": 0, "prefix_tree/raw_tokens": 0}

    total_raw = sum(len(s) for s in sequences)
    if total_raw == 0:
        return {"prefix_tree/sharing_ratio": 0.0, "prefix_tree/flat_tokens": 0, "prefix_tree/raw_tokens": 0}

    _, num_tokens = greedy_build_tries(sequences, max_tokens_per_tree=total_raw * 10)
    flat = sum(num_tokens)
    return {
        "prefix_tree/sharing_ratio": 1.0 - flat / total_raw,
        "prefix_tree/flat_tokens": flat,
        "prefix_tree/raw_tokens": total_raw,
    }


# ============================================================================
# Micro-batch preparation (consumed by engine utils)
# ============================================================================


def prepare_prefix_tree_micro_batches(
    data,
    sp_size: int,
    dp_group=None,
    same_micro_num_in_dp: bool = True,
    num_batches_divided_by: int | None = None,
):
    """Prepare micro-batches using prefix-tree DFS grouping.

    Expects a pre-built trie stored via ``tu.set_non_tensor_data(data,
    "prefix_tree", trie)``. If not present, builds one from sequences
    (backward-compatible fallback).
    """
    import logging as _logging

    import torch

    from verl.utils import tensordict_utils as tu

    _logging.getLogger(__name__).warning_once(
        "prefix_tree is on: max_token_len_per_gpu is interpreted as "
        "deduplicated (flat trie) token count, not raw sequence length."
    )

    assert "max_token_len_per_gpu" in data.keys(), "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
    max_token_len_per_gpu = data["max_token_len_per_gpu"]
    max_token_len = max_token_len_per_gpu * sp_size

    trie = tu.get_non_tensor_data(data, "prefix_tree", default=None)
    if trie is not None:
        batch_idx_list = mbs_groups_from_trie(trie, max_token_len)
    else:
        input_ids = data["input_ids"]
        seqs = [t.tolist() for t in input_ids.unbind()]
        batch_idx_list = dfs_micro_batch_groups(seqs, max_token_len)

    if torch.distributed.is_initialized() and same_micro_num_in_dp and dp_group is not None:
        from verl.utils.device import get_torch_device

        n_mb = torch.tensor([len(batch_idx_list)], device=get_torch_device().current_device())
        torch.distributed.all_reduce(n_mb, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        while len(batch_idx_list) < n_mb.item():
            batch_idx_list.append(batch_idx_list[-1])

    if num_batches_divided_by is not None:
        from verl.utils.seqlen_balancing import roundup_divisible

        target = roundup_divisible(len(batch_idx_list), num_batches_divided_by)
        while len(batch_idx_list) < target:
            batch_idx_list.append(batch_idx_list[-1])

    micro_batches = [tu.index_select_tensor_dict(data, idx) for idx in batch_idx_list]
    return micro_batches, batch_idx_list


# ============================================================================
# Load balancing (consumed by trainers)
# ============================================================================


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

    import torch

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
        from verl.utils import tensordict_utils as tu

        data = tu.index_select_tensor_dict(data, torch.tensor(dfs_order))

    if hasattr(data, "batch") and "attention_mask" in data.batch:
        global_seqlen_lst = data.batch["attention_mask"].view(batch_size, -1).sum(-1)
    else:
        global_seqlen_lst = torch.Tensor([item.size()[0] for item in data["input_ids"]])

    if contiguous_partitions:
        per_rank = batch_size // dp_size
        partition_lst = [list(range(i * per_rank, (i + 1) * per_rank)) for i in range(dp_size)]
    else:
        from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions

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

    import torch

    from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions

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

    import torch

    from verl.utils.seqlen_balancing import log_seqlen_unbalance

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
