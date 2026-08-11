"""End-to-end test: segment metadata -> global trie -> leaf_idx propagation.

Tests the wiring contract of ray_trainer._build_global_trie without importing
verl.protocol (heavy deps). The local replica below mirrors production logic:

- build_global_tree_from_segments attaches a trie whose leaves map to every sample
- leaf_idx (torch.long tensor) is valid for torch-index reorder/chunk semantics
- fallback to greedy_build_tries when segment metadata is absent
- multilevel segments produce ancestor + intermediate + leaf nodes at each level
"""

from __future__ import annotations

import numpy as np
import torch

from verl.utils.prefix_tree.segment_grouper import create_grpo_segment_metadata, create_segment_metadata
from verl.utils.prefix_tree.tree import TrieNode, build_global_tree_from_segments


def _build_global_trie_local(seqs_t, seg_hashes=None, seg_lengths=None):
    """Returns (trie, leaf_idx) mirroring ray_trainer._build_global_trie."""
    total_raw = sum(int(s.numel()) for s in seqs_t)
    trie = None
    if seg_hashes is not None and seg_lengths is not None:
        trie = build_global_tree_from_segments(seqs_t, seg_hashes, seg_lengths)
    if trie is None:
        from verl.utils.prefix_tree.tree import PrefixTrie

        trie = PrefixTrie(root=TrieNode())
        for seq_id, seq in enumerate(seqs_t):
            trie.insert(np.array(seq if hasattr(seq, "tolist") else [int(x) for x in seq], dtype=np.int64), seq_id)
        trie.finalize()
        if total_raw <= 0:
            trie = None
    if trie is None:
        return None, None

    leaf_idx = np.full(len(seqs_t), -1, dtype=np.int64)
    for node_idx, node in enumerate(trie.nodes):
        if not node.children:
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node_idx
    return trie, torch.from_numpy(leaf_idx)


def _make_seqs(n_prompts=2, rollout_n=2, prompt_len=5, resp_len=3):
    seqs, uids = [], []
    for p in range(n_prompts):
        prompt = list(range(100 + p * 10, 100 + p * 10 + prompt_len))
        for r in range(rollout_n):
            resp = list(range(200 + (p * rollout_n + r) * 10, 200 + (p * rollout_n + r) * 10 + resp_len))
            seqs.append(torch.tensor(prompt + resp, dtype=torch.long))
            uids.append(f"p{p}")
    seg_hashes, seg_lengths = create_grpo_segment_metadata(uids, [prompt_len] * len(seqs), rollout_n)
    return seqs, seg_hashes, seg_lengths


def _make_multilevel_seqs():
    # 2 prompts x 2 rollouts; layout: [turn1(shared)|turn2(shared)|response(unique)]
    seqs = [
        torch.tensor(t, dtype=torch.long)
        for t in [
            [10, 11, 12, 20, 21, 30, 31],
            [10, 11, 12, 20, 21, 32, 33],
            [50, 51, 52, 60, 61, 70, 71],
            [50, 51, 52, 60, 61, 72, 73],
        ]
    ]
    segs = [[(f"p{i // 2}_t1", 3), (f"p{i // 2}_t2", 2)] for i in range(4)]
    return (seqs,) + create_segment_metadata(segs)


def test_build_global_trie_leaf_idx_valid_and_fallback():
    seqs, seg_hashes, seg_lengths = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs, seg_hashes, seg_lengths)
    assert trie is not None and len(trie.nodes) > 0
    assert isinstance(leaf_idx, torch.Tensor) and leaf_idx.dtype == torch.long
    assert leaf_idx.shape == (4,) and (leaf_idx >= 0).all()
    for i, node_idx in enumerate(leaf_idx.tolist()):
        node = trie.nodes[node_idx]
        assert not node.children and i in node.sequence_ids
    # Fallback to greedy when no segment metadata
    trie_fb, leaf_fb = _build_global_trie_local(seqs, seg_hashes=None, seg_lengths=None)
    assert trie_fb is not None and leaf_fb is not None and (leaf_fb >= 0).all()


def test_multilevel_builds_three_node_levels():
    seqs, seg_hashes, seg_lengths = _make_multilevel_seqs()
    trie = build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)
    assert trie is not None and len(trie.nodes) == 8  # 2 prefix + 2 inter + 4 leaves
    leaves = [n for n in trie.nodes if not n.children]
    intermediates = [n for n in trie.nodes if n.children and n.ancestor is not None]
    roots = [n for n in trie.nodes if n.ancestor is None]
    assert len(leaves) == 4 and len(intermediates) == 2 and len(roots) == 2
    assert all(leaf.ancestor in intermediates for leaf in leaves)
    # Per-level input_ids: turn1 roots, turn2 intermediates, response leaves
    assert {(10, 11, 12), (50, 51, 52)} == {tuple(int(x) for x in n.input_ids) for n in roots}
    assert list(trie.leaves[0].input_ids) == [30, 31] and list(trie.leaves[3].input_ids) == [72, 73]
