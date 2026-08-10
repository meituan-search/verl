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
"""Prefix-tree data structures.

Class hierarchy:

    TrieNode:       compressed trie node; root holds flat DFS-ordered ``nodes``
                      list; ``nodes[i].node_idx == i`` for O(1) lookup and future
                      KV-cache indexing.

    PrefixTrie:     common interface for navigating a prefix trie.
                      Both the global trie and per-micro-batch view expose this
                      interface so callers need not distinguish between them.
                      Future: ``kv_cache: list[Tensor]`` indexed by ``node_idx``.

    PrefixSubTrie:  per-micro-batch view into a PrefixTrie.  Inherits the
                      PrefixTrie interface.  Serialisable: ``leaf_node_ids`` and
                      ``leaf_to_sample`` are plain int lists; ``source`` is a
                      local-only back-reference, never transmitted cross-node.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# TrieNode: compressed trie node (mutable before finalize, immutable after)
# ---------------------------------------------------------------------------


@dataclass
class TrieNode:
    """Compressed-trie node.

    Each non-root node represents a contiguous run of tokens shared by the same
    set of sequences.  The root has no tokens and no parent (``ancestor=None``).

    Build phase (before ``finalize()``): ``input_ids``, ``children``,
    ``sequence_ids``, ``ancestor`` are set by insert/split.

    After ``finalize()``: ``node_idx``, ``nodes``, ``leaves`` are assigned once
    and the tree becomes immutable.

    ``nodes`` (root-only): flat DFS-ordered list of all non-root nodes.
    ``nodes[i].node_idx == i`` for O(1) lookup.
    """

    input_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    children: dict[int, TrieNode] = field(default_factory=dict)
    # Sequence IDs that pass through this node (from trie construction).
    sequence_ids: list[int] = field(default_factory=list)
    # Direct parent reference: single hop upward; None on root.
    ancestor: Optional[TrieNode] = None
    # Root-only: flat DFS list; nodes[i].node_idx == i.
    nodes: list[TrieNode] = field(default_factory=list)
    # Root-only: leaves[sample_idx] = leaf TrieNode for that sample.
    # Direct O(1) index → leaf map; populated during finalize().
    leaves: list[Optional[TrieNode]] = field(default_factory=list)
    # DFS index in root.nodes: -1 until finalize().
    node_idx: int = -1

    @property
    def is_root(self) -> bool:
        return self.node_idx == -1

    def _add_child(self, input_ids: np.ndarray, sequence_ids: list[int] | None = None) -> TrieNode:
        """Add a new child node. Must only be called before ``finalize()``."""
        child = TrieNode(
            input_ids=input_ids,
            sequence_ids=sequence_ids or [],
            ancestor=self,
        )
        self.children[int(input_ids[0])] = child
        return child

    def insert(self, sequence: np.ndarray, seq_id: int) -> None:
        """Insert a full token sequence. Must only be called before ``finalize()``.

        Walks the trie matching prefixes. On full match, descends into child.
        On partial match, splits the child and adds a divergent branch. On no
        match, adds a new child. seq_id is appended to every node on the path.
        """
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
        """Split this node's run at ``match_pos`` into prefix + suffix.

        ``self`` keeps ``input_ids[:match_pos]`` (the shared prefix) and becomes
        the parent.  A new suffix node is created for ``input_ids[match_pos:]``,
        inheriting the old children, sequence_ids, and ancestor (for leaf nodes).

        Only the child link and input_ids are modified — no re-allocation of
        the node itself.  Must only be called before ``finalize()``.

        Returns the suffix node.
        """
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

    def finalize(self) -> None:
        """Assign node_idx, populate nodes and leaves (root-only).

        Walks the tree in DFS pre-order (sorted by first token for deterministic,
        insertion-order-independent output), assigning ``node_idx`` to each
        node and collecting them into root's ``nodes`` list.

        Also re-sorts each node's ``children`` dict to match the DFS order, so
        that ``children.values()`` yields the same order as ``trie.nodes`` —
        all downstream code gets one consistent traversal regardless of
        whether it iterates ``trie.nodes`` or walks ``children``.

        After finalize, the tree is immutable — do not call ``insert()`` or
        ``split()``.
        """
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
        self.children = dict(sorted(self.children.items()))
        for child in self.children.values():
            _walk(child, None)


# ---------------------------------------------------------------------------
# PrefixTrie: common interface (global trie + subtrie view)
# ---------------------------------------------------------------------------


class PrefixTrie:
    """Common interface for navigating and building a prefix trie.

    Concrete subclasses:
      - PrefixTrie itself:  wraps the global TrieNode root for a full batch
      - PrefixSubTrie:      filtered view for one micro-batch

    Both expose the same interface so downstream code (layout builder, MAGI
    key construction, KV-cache lookup) works identically on either.
    """

    nodes: list[TrieNode]  # flat DFS list; nodes[i].node_idx == i
    root: TrieNode  # virtual root (is_root=True)

    def __init__(self, root: TrieNode | None = None) -> None:
        if root is None:
            root = TrieNode()  # empty root for building
        self.root = root
        self.nodes = root.nodes  # shared reference, no copy

    # ── navigation ────────────────────────────────────────────────────────

    def __getitem__(self, node_idx: int) -> TrieNode:
        """O(1) node lookup by node_idx."""
        return self.nodes[node_idx]

    def __iter__(self):
        """Iterate all nodes in DFS order (same as trie.nodes)."""
        return iter(self.nodes)

    @property
    def leaf_nodes(self) -> list[TrieNode]:
        """Leaf nodes (no children) in DFS order from trie.nodes.

        Single source of truth — replaces per-consumer tree walking.
        """
        return [n for n in self.nodes if not n.children]

    # ── build ─────────────────────────────────────────────────────────────

    @staticmethod
    def _unfinalized(method):
        """Decorator: raise if the tree is finalized."""

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if self.nodes:
                raise RuntimeError(f"{method.__name__}() called after finalize() — tree is immutable.")
            return method(self, *args, **kwargs)

        return wrapper

    @_unfinalized
    def insert(self, sequence, seq_id: int) -> None:
        """Insert a full token sequence. Must be called before finalize()."""
        self.root.insert(sequence, seq_id)

    def finalize(self) -> None:
        """Assign node_idx, populate nodes and leaves. Tree is immutable after."""
        self.root.finalize()
        self.nodes = self.root.nodes  # update shared reference

    # ── metrics ───────────────────────────────────────────────────────────

    # ── future: KV cache ──────────────────────────────────────────────────
    # kv_cache: list[Optional[Tensor]]  # indexed by node_idx, same as nodes


# ---------------------------------------------------------------------------
# PrefixSubTrie: per-micro-batch view
# ---------------------------------------------------------------------------


class PrefixSubTrie(PrefixTrie):
    """Per-micro-batch view into a PrefixTrie.

    Exposes the same PrefixTrie interface so callers treat it identically to
    the global trie.  ``nodes`` contains only the TrieNodes relevant to this
    micro-batch (leaves and their ancestor path).

    Serialisation
    -------------
    ``leaf_node_ids`` and ``leaf_to_sample`` are plain int lists (tiny cross-
    node footprint).  ``source`` is a local-only reference (not serialised);
    re-attach after deserialisation to enable KV-cache lookup.
    """

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
        self.nodes = self._collect_nodes(source, leaf_node_ids)
        # Build shard-local leaf_ids: indexed by local position (0..shard_size-1).
        # global_sample_ids[i] is the global sample index for local position i.
        self.global_sample_ids = sorted(set(leaf_to_sample))
        _global_to_local: dict[int, int] = {g: i for i, g in enumerate(self.global_sample_ids)}
        local_batch_size = len(self.global_sample_ids)
        self.leaf_ids = np.full(local_batch_size, -1, dtype=np.int64)
        for i, sample_idx in enumerate(leaf_to_sample):
            self.leaf_ids[_global_to_local[sample_idx]] = leaf_node_ids[i]

    @staticmethod
    def _collect_nodes(source: PrefixTrie, leaf_node_ids: list[int]) -> list[TrieNode]:
        """Collect nodes reachable from the given leaves (leaves + all ancestors)."""
        seen: set[int] = set()
        result: list[TrieNode] = []
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
                    result.append(n)
        return result

    def __getstate__(self) -> dict:
        """Pickle without the full trie back-references.

        TrieNode.children dicts point to siblings outside the subtrie, causing
        pickle to serialise the entire global trie.  Store compact per-node data
        and reconstruct detached nodes on __setstate__.
        """
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
            node = TrieNode(
                input_ids=np.array(input_ids, dtype=np.int64), sequence_ids=list(sequence_ids), node_idx=node_idx
            )
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


# ---------------------------------------------------------------------------
# Segment-based global trie construction (O(N), no token comparison)
# ---------------------------------------------------------------------------


def build_global_tree_from_segments(
    samples: list,
    segment_hashes,
    segment_lengths,
) -> Optional[TrieNode]:
    """Build a global TrieNode from segment metadata, supporting multilevel trees.

    O(N) construction using known prefix structure, no token-by-token
    comparison.  Returns a TrieNode root compatible with ``mbs_groups_from_trie``
    and ``subtrie_view``.

    Recurses through every segment level: samples sharing levels 0..k get an
    ancestor node at each shared level.  A sample bottoms out into a leaf when
    its segments are exhausted, when its subgroup becomes a singleton, or when
    no further sharing is found.  Leaf ``input_ids`` = all remaining tokens
    after the last shared segment.

    Children of the root are keyed by level-0 hash (deterministic DFS order).
    Children of intermediate nodes are keyed by ``node_idx`` (unique).

    Args:
        samples: List of 1-D token tensors or lists (one per sequence).
        segment_hashes: (N,) object-dtype numpy array of hash lists.
        segment_lengths: (N,) object-dtype numpy array of length lists.

    Returns:
        TrieNode root, or None if fewer than 2 samples.
    """
    if not samples or len(samples) < 2:
        return None

    from verl.utils.prefix_tree.segment_grouper import group_by_segment_hash

    # Normalize samples to list[list[int]]: production caller passes lists,
    # tests may pass tensors.
    samples = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]

    groups = group_by_segment_hash(segment_hashes, segment_lengths, level=0)

    trie_root = TrieNode()
    trie_root.nodes = []
    trie_root.leaves = [None] * len(samples)

    for uid_hash in sorted(groups.keys()):
        group = groups[uid_hash]
        all_seq_ids = [sid for sid, _ in group]

        if len(group) >= 2:
            first_idx, prefix_len = group[0]
            prefix_tokens = samples[first_idx][:prefix_len]

            prefix_node = TrieNode(
                input_ids=prefix_tokens,
                sequence_ids=list(all_seq_ids),
                ancestor=None,
                node_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(prefix_node)
            trie_root.children[uid_hash] = prefix_node

            _build_segment_subtree(
                samples,
                segment_hashes,
                segment_lengths,
                group_sids=all_seq_ids,
                level=1,
                accumulated_len=prefix_len,
                parent_node=prefix_node,
                trie_root=trie_root,
            )
        else:
            seq_idx = group[0][0]
            all_tokens = samples[seq_idx]
            leaf = TrieNode(
                input_ids=all_tokens,
                sequence_ids=[seq_idx],
                ancestor=None,
                node_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(leaf)
            trie_root.children[uid_hash] = leaf
            trie_root.leaves[seq_idx] = leaf

    return trie_root if trie_root.nodes else None


def _build_segment_subtree(
    samples,
    segment_hashes,
    segment_lengths,
    group_sids: list[int],
    level: int,
    accumulated_len: int,
    parent_node: TrieNode,
    trie_root: TrieNode,
) -> None:
    """Recursively build ancestor nodes for shared segments at ``level`` >= 1.

    Samples in *group_sids* already share levels 0..level-1.  Subgroup by hash
    at *level*: 2+ sharing → ancestor node + recurse on the suffix; singleton
    subgroup or segment-exhausted sample → leaf with all remaining tokens.
    """
    # Samples whose segment list ended before this level → leaves with remaining tokens.
    for sid in group_sids:
        if level >= len(segment_hashes[sid]):
            remaining = samples[sid][accumulated_len:]
            leaf = TrieNode(
                input_ids=remaining,
                sequence_ids=[sid],
                ancestor=parent_node,
                node_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(leaf)
            parent_node.children[leaf.node_idx] = leaf
            trie_root.leaves[sid] = leaf

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
                input_ids=seg_tokens,
                sequence_ids=list(subgroup),
                ancestor=parent_node,
                node_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(node)
            parent_node.children[node.node_idx] = node
            _build_segment_subtree(
                samples,
                segment_hashes,
                segment_lengths,
                group_sids=subgroup,
                level=level + 1,
                accumulated_len=accumulated_len + seg_len,
                parent_node=node,
                trie_root=trie_root,
            )
        else:
            sid = subgroup[0]
            remaining = samples[sid][accumulated_len:]
            leaf = TrieNode(
                input_ids=remaining,
                sequence_ids=[sid],
                ancestor=parent_node,
                node_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(leaf)
            parent_node.children[leaf.node_idx] = leaf
            trie_root.leaves[sid] = leaf


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
