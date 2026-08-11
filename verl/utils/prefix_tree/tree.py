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
"""Prefix-tree data structures: TrieNode (compressed trie), PrefixTrie (full batch), PrefixSubTrie (per-mb serializable view), segment-based tree builder."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# TrieNode: compressed trie node (mutable before finalize, immutable after).


@dataclass(eq=False)
class TrieNode:
    """Compressed-trie node with input_ids, children, sequence_ids, ancestor. No tree-level metadata (managed by PrefixTrie).

    Transient layout attrs set externally by build_layout_from_tree_node: _flat_start, _flat_end, _owner_offset, _owner_sample."""

    input_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    children: dict[int, TrieNode] = field(default_factory=dict)
    # Sequence IDs that pass through this node (from trie construction).
    sequence_ids: list[int] = field(default_factory=list)
    # Direct parent reference: single hop upward; None on root.
    ancestor: Optional[TrieNode] = None

    @property
    def is_root(self) -> bool:
        return self.ancestor is None

    def _add_child(self, input_ids: np.ndarray, sequence_ids: list[int] | None = None) -> TrieNode:
        """Add a new child node. Must only be called before finalize()."""
        child = TrieNode(
            input_ids=input_ids,
            sequence_ids=sequence_ids or [],
            ancestor=self,
        )
        self.children[int(input_ids[0])] = child
        return child

    def insert(self, sequence: np.ndarray, seq_id: int) -> None:
        """Insert token sequence: match prefix, split child on partial match, add divergent branch. Before finalize()."""
        if len(sequence) == 0:
            return
        token = int(sequence[0])
        if token not in self.children:
            self._add_child(sequence, [seq_id])
            return
        child = self.children[token]
        run = child.input_ids
        n = min(len(run), len(sequence))
        mismatches = np.where(run[:n] != sequence[:n])[0]
        match = int(mismatches[0]) if len(mismatches) > 0 else n
        if match == len(run):
            # Full match — append seq_id, recurse on remainder.
            child.sequence_ids.append(seq_id)
            child.insert(sequence[match:], seq_id)
        else:
            # Partial match — split, add divergent branch.
            child._split(match)
            child.sequence_ids.append(seq_id)
            remaining = sequence[match:]
            if len(remaining) > 0:
                child._add_child(remaining, [seq_id])

    def _split(self, match_pos: int) -> TrieNode:
        """Split node at match_pos into prefix (self) + suffix (new child inheriting old children/seq_ids). Returns suffix."""
        old_run = self.input_ids
        old_children = self.children
        old_seq_ids = self.sequence_ids

        # Suffix: inherits old children, COPY of old seq_ids.
        suffix = TrieNode(
            input_ids=old_run[match_pos:].copy(),
            children=old_children,
            sequence_ids=list(old_seq_ids),
            ancestor=self,
        )
        # Prefix: this node keeps the shared part, resets children.
        self.input_ids = old_run[:match_pos].copy()
        self.children = {int(old_run[match_pos]): suffix}
        # sequence_ids stays (prefix node keeps its seq_ids + caller appends).
        return suffix


# PrefixTrie: common interface for navigating a prefix trie (global + subtrie views).


class PrefixTrie:
    """Common interface for prefix trie navigation/building. Classes: PrefixTrie (global), PrefixSubTrie (per-mb).

    Manages nodes, leaves, _finalized, node_idx assignment. Same API for both."""

    def __init__(self, root: TrieNode | None = None) -> None:
        if root is None:
            root = TrieNode()  # empty root for building
        self.root = root
        self.nodes: list[TrieNode] = []
        self.leaves: list[Optional[TrieNode]] = []
        self._finalized: bool = False

    # ── root delegation (backward compat for callers that use trie.children) ─

    @property
    def children(self) -> dict[int, TrieNode]:
        """Delegate to root's children so callers can use trie.children."""
        return self.root.children

    @property
    def is_root(self) -> bool:
        """Always True — PrefixTrie wraps the root node."""
        return True

    # ── navigation ────────────────────────────────────────────────────────

    def __getitem__(self, node_idx: int) -> TrieNode:
        """O(1) node lookup by node_idx."""
        return self.nodes[node_idx]

    def __iter__(self):
        """Iterate all nodes in DFS order."""
        return iter(self.nodes)

    @property
    def leaf_nodes(self) -> list[TrieNode]:
        """Leaf nodes (no children) in DFS order."""
        return [n for n in self.nodes if not n.children]

    # ── build ─────────────────────────────────────────────────────────────

    @staticmethod
    def _unfinalized(method):
        """Decorator: raise if the tree is finalized."""

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._finalized:
                raise RuntimeError(f"{method.__name__}() called after finalize() — tree is immutable.")
            return method(self, *args, **kwargs)

        return wrapper

    @_unfinalized
    def insert(self, sequence, seq_id: int) -> None:
        """Insert a full token sequence. Must be called before finalize()."""
        self.root.insert(sequence, seq_id)

    def finalize(self) -> None:
        """Assign node_idx via DFS pre-order (sorted by first token), populate nodes/leaves. Immutable after."""
        self.nodes = []
        self.leaves = []

        def _walk(node: TrieNode, parent: Optional[TrieNode]) -> None:
            node.node_idx = len(self.nodes)
            node.ancestor = parent
            self.nodes.append(node)
            if not node.children:
                for sid in node.sequence_ids:
                    while len(self.leaves) <= sid:
                        self.leaves.append(None)
                    self.leaves[sid] = node
            else:
                # Sort children by first token so DFS order is deterministic
                # and matches trie.nodes — single source of truth for traversal.
                node.children = dict(sorted(node.children.items()))
                for child in node.children.values():
                    _walk(child, node)

        # Root's children: also sort for consistency.
        self.root.children = dict(sorted(self.root.children.items()))
        for child in self.root.children.values():
            _walk(child, None)

        self._finalized = True

    # ── metrics ───────────────────────────────────────────────────────────

    # ── future: KV cache ──────────────────────────────────────────────────
    # kv_cache: list[Optional[Tensor]]  # indexed by node_idx, same as nodes


# PrefixSubTrie: per-micro-batch view.


class PrefixSubTrie(PrefixTrie):
    """Per-micro-batch PrefixTrie view with same interface. Serialisable via __getstate__/__setstate__.

    leaf_to_sample: leaf i → global sample index
    leaf_node_ids: leaf i → node_idx in source
    source: back-ref to global trie (not serialised)
    leaf_ids: shard-local position → leaf node_idx (-1 if absent)
    global_sample_ids: shard-local position → global sample index"""

    leaf_to_sample: list[int]  # leaf i → global sample index
    leaf_node_ids: list[int]  # leaf i → node_idx of its TrieNode in source
    source: Optional[PrefixTrie]  # back-ref to global trie; not serialised
    # leaf_ids[local_pos] = node_idx of that sample's leaf; -1 if absent.
    # Indexed by shard-local position (0..len(global_sample_ids)-1).
    # Use global_sample_ids[local_pos] to recover the global sample index.
    leaf_ids: np.ndarray  # shape (len(global_sample_ids),), dtype int64
    # global_sample_ids[i] = global sample index for shard-local position i.
    global_sample_ids: list[int]

    # MAGI key cache for OLP→actor_update reuse. Populated on first forward,
    # reused if tree structure and CP group haven't changed.
    _cached_magi_key: Optional[object] = None

    def __init__(
        self,
        source: PrefixTrie,
        leaf_node_ids: list[int],
        leaf_to_sample: list[int],
        batch_size: int,
    ) -> None:
        import numpy as np

        self.source = source
        self.leaf_node_ids = leaf_node_ids
        self.leaf_to_sample = leaf_to_sample
        self.root = source.root
        self._finalized = True  # subtries are always finalized (read-only views)
        self.nodes, self.leaves = self._collect_nodes(source, leaf_node_ids)
        # Build shard-local leaf_ids: indexed by local position (0..shard_size-1).
        # global_sample_ids[i] is the global sample index for local position i.
        self.global_sample_ids = sorted(set(leaf_to_sample))
        _global_to_local: dict[int, int] = {g: i for i, g in enumerate(self.global_sample_ids)}
        local_batch_size = len(self.global_sample_ids)
        self.leaf_ids = np.full(local_batch_size, -1, dtype=np.int64)
        for i, sample_idx in enumerate(leaf_to_sample):
            self.leaf_ids[_global_to_local[sample_idx]] = leaf_node_ids[i]

    @staticmethod
    def _collect_nodes(source: PrefixTrie, leaf_node_ids: list[int]) -> tuple[list[TrieNode], list[Optional[TrieNode]]]:
        """Collect leaves + all ancestors reachable from given leaf_node_ids.

        Returns nodes in DFS order with original (global) node_idx preserved.
        Builds a _node_idx_to_local remapping for subtrie-local indexing.
        """
        seen: set[int] = set()
        nodes: list[TrieNode] = []
        for idx in leaf_node_ids:
            node = source[idx]
            path: list[TrieNode] = []
            cur: Optional[TrieNode] = node
            while cur is not None and cur.node_idx not in seen:
                path.append(cur)
                cur = cur.ancestor
            for n in reversed(path):
                if n.node_idx not in seen:
                    seen.add(n.node_idx)
                    nodes.append(n)

        # Build leaves indexed by sequence_id (global indexing).
        leaves: list[Optional[TrieNode]] = []
        for node in nodes:
            if not node.children:
                for sid in node.sequence_ids:
                    while len(leaves) <= sid:
                        leaves.append(None)
                    leaves[sid] = node
        return nodes, leaves

    def __getstate__(self) -> dict:
        """Pickle compactly: store per-node data (node_idx, input_ids, ancestor, seq_ids) to avoid full trie serialisation."""
        nodes_data = [
            (n.node_idx, n.input_ids, n.ancestor.node_idx if n.ancestor else -1, n.sequence_ids) for n in self.nodes
        ]
        return {
            "leaf_node_ids": self.leaf_node_ids,
            "leaf_to_sample": self.leaf_to_sample,
            "leaf_ids": self.leaf_ids,
            "global_sample_ids": self.global_sample_ids,
            "nodes_data": nodes_data,
        }

    def __setstate__(self, state: dict) -> None:
        self.source = None
        self.root = TrieNode()  # dummy root (not used after unpickling)
        self.leaf_node_ids = state["leaf_node_ids"]
        self.leaf_to_sample = state["leaf_to_sample"]
        self.leaf_ids = state["leaf_ids"]
        self.global_sample_ids = state.get("global_sample_ids", sorted(set(state["leaf_to_sample"])))
        # Reconstruct detached TrieNode objects (subtrie-only children links).
        by_node_idx: dict[int, TrieNode] = {}
        for node_idx, input_ids, _anc, sequence_ids in state["nodes_data"]:
            node = TrieNode(input_ids=np.array(input_ids, dtype=np.int64), sequence_ids=list(sequence_ids))
            node.node_idx = node_idx
            by_node_idx[node_idx] = node

        for node_idx, input_ids, ancestor_node_idx, _seq in state["nodes_data"]:
            node = by_node_idx[node_idx]
            if ancestor_node_idx != -1 and ancestor_node_idx in by_node_idx:
                node.ancestor = by_node_idx[ancestor_node_idx]
                # Key by node_idx (always unique) not first_token: keying by first
                # token causes silent collision when two siblings start with the same
                # token (e.g. two rollout responses both begin with the same word).
                by_node_idx[ancestor_node_idx].children[node_idx] = node

        self.nodes = [by_node_idx[fid] for fid, _, _, _ in state["nodes_data"]]
        self._cached_magi_key = None


# Segment-based global trie construction (O(N), no token comparison).


def build_global_tree_from_segments(
    samples: list,
    segment_hashes,
    segment_lengths,
) -> Optional[PrefixTrie]:
    """Build global PrefixTrie from segment metadata (O(N), multilevel). Samples share segments with matching hashes.

    Leaf input_ids = all remaining tokens after last shared segment. Compatible with mbs_groups_from_trie/subtrie_view."""
    if not samples or len(samples) < 2:
        return None

    from verl.utils.prefix_tree.segment_grouper import group_by_segment_hash

    # Normalize samples to list[list[int]]: production caller passes lists,
    # tests may pass tensors.
    samples = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]

    groups = group_by_segment_hash(segment_hashes, segment_lengths, level=0)

    trie_root = TrieNode()
    _key_counter = [0]

    def _next_key() -> int:
        _key_counter[0] += 1
        return _key_counter[0]

    for uid_hash in sorted(groups.keys()):
        group = groups[uid_hash]
        all_seq_ids = [sid for sid, _ in group]

        if len(group) >= 2:
            first_idx, prefix_len = group[0]
            prefix_tokens = samples[first_idx][:prefix_len]

            prefix_node = TrieNode(
                input_ids=np.array(prefix_tokens),
                sequence_ids=list(all_seq_ids),
                ancestor=None,
            )
            trie_root.children[uid_hash] = prefix_node

            _build_segment_subtree(
                samples,
                segment_hashes,
                segment_lengths,
                group_sids=all_seq_ids,
                level=1,
                accumulated_len=prefix_len,
                parent_node=prefix_node,
                next_key=_next_key,
            )
        else:
            seq_idx = group[0][0]
            all_tokens = samples[seq_idx]
            leaf = TrieNode(
                input_ids=np.array(all_tokens),
                sequence_ids=[seq_idx],
                ancestor=None,
            )
            trie_root.children[uid_hash] = leaf

    if not trie_root.children:
        return None

    trie = PrefixTrie(root=trie_root)
    trie.finalize()
    return trie


def _build_segment_subtree(
    samples,
    segment_hashes,
    segment_lengths,
    group_sids: list[int],
    level: int,
    accumulated_len: int,
    parent_node: TrieNode,
    next_key,
) -> None:
    """Recursively build ancestor nodes for shared segments at level >= 1: subgroup by hash, create ancestor/leaf nodes."""
    # Samples whose segment list ended before this level → leaves with remaining tokens.
    for sid in group_sids:
        if level >= len(segment_hashes[sid]):
            remaining = samples[sid][accumulated_len:]
            leaf = TrieNode(
                input_ids=np.array(remaining),
                sequence_ids=[sid],
                ancestor=parent_node,
            )
            parent_node.children[next_key()] = leaf

    # Subgroup remaining samples by hash at this level.
    subgroups: dict[int, list[int]] = {}
    for sid in group_sids:
        if level < len(segment_hashes[sid]):
            h = int(segment_hashes[sid][level])
            subgroups.setdefault(h, []).append(sid)

    for hash_val in sorted(subgroups.keys()):
        subgroup = subgroups[hash_val]
        if len(subgroup) >= 2:
            seg_len = int(segment_lengths[subgroup[0]][level])
            seg_tokens = samples[subgroup[0]][accumulated_len : accumulated_len + seg_len]
            node = TrieNode(
                input_ids=np.array(seg_tokens),
                sequence_ids=list(subgroup),
                ancestor=parent_node,
            )
            parent_node.children[next_key()] = node
            _build_segment_subtree(
                samples,
                segment_hashes,
                segment_lengths,
                group_sids=subgroup,
                level=level + 1,
                accumulated_len=accumulated_len + seg_len,
                parent_node=node,
                next_key=next_key,
            )
        else:
            sid = subgroup[0]
            remaining = samples[sid][accumulated_len:]
            leaf = TrieNode(
                input_ids=np.array(remaining),
                sequence_ids=[sid],
                ancestor=parent_node,
            )
            parent_node.children[next_key()] = leaf


def trie_ancestors(node: TrieNode) -> list[TrieNode]:
    """Return ancestor chain from root-child down to node's parent (exclusive of node)."""
    chain: list[TrieNode] = []
    cur = node.ancestor
    while cur is not None:
        chain.append(cur)
        cur = cur.ancestor
    chain.reverse()
    return chain


def _is_prefix_tree_enabled(config_or_data) -> bool:
    if isinstance(config_or_data, dict):
        return config_or_data.get("use_prefix_tree", False)
    return bool(getattr(config_or_data, "use_prefix_tree", False))
