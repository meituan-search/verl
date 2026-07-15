#!/usr/bin/env python3
"""End-to-end test: segment metadata → global trie → leaf_idx propagation.

Tests the wiring contract of ray_trainer._build_global_trie without importing
verl.protocol (which has heavy deps). Uses direct module loading + stub pattern
matching test_mbs_uid.py.

Verifies:
- _build_global_trie attaches trie (TrieNode root) and leaf_idx (np.int64 array)
- leaf_idx survives numpy fancy-index reordering (the contract DataProto.reorder uses)
- each leaf_idx[i] points to a valid leaf node (no children) in the trie
- fast path falls back to greedy_build_tries when segment metadata absent
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import torch

# ── Direct module loading (bypasses __init__.py) ─────────────────────────────
_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "utils", "prefix_tree"))
for pkg_name in ["verl", "verl.utils", "verl.utils.prefix_tree"]:
    if pkg_name not in sys.modules:
        m = types.ModuleType(pkg_name)
        m.__path__ = []
        sys.modules[pkg_name] = m

# Stub verl.utils.tensordict_utils (only the symbols segment_grouper/tree need)
_tu = types.ModuleType("verl.utils.tensordict_utils")
_tu.get_non_tensor_data = lambda *a, **k: None
_tu.assign_non_tensor = lambda *a, **k: None
sys.modules["verl.utils.tensordict_utils"] = _tu

# Stub verl.utils.device (dynamic.py imports get_torch_device)
_dev = types.ModuleType("verl.utils.device")
_dev.get_torch_device = lambda: type("D", (), {"current_device": lambda s: "cpu"})()
sys.modules["verl.utils.device"] = _dev

# Stub verl.utils.seqlen_balancing (dynamic.py imports 4 helpers from it).
# Matches the stub pattern in test_mbs_uid.py — KK-style greedy balancer.
_seq = types.ModuleType("verl.utils.seqlen_balancing")
_seq.calculate_workload = lambda x: 24576 * x + x**2
_seq.log_seqlen_unbalance = lambda *a, **k: None
_seq.roundup_divisible = lambda a, b: (a + b - 1) // b


def _kk(seqlen_list, k_partitions, equal_size=False):
    """Simple Karmarkar-Karp stub: largest-differencing greedy."""
    items = sorted([(s, i) for i, s in enumerate(seqlen_list)], reverse=True)
    partitions = [[] for _ in range(k_partitions)]
    sums = [0] * k_partitions
    for s, i in items:
        min_idx = sums.index(min(sums))
        partitions[min_idx].append(i)
        sums[min_idx] += s
    return partitions


_seq.get_seqlen_balanced_partitions = _kk
sys.modules["verl.utils.seqlen_balancing"] = _seq


def _load_module(name: str, rel_path: str):
    """Load a module directly from file path, registering in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load prefix-tree modules directly (skip package __init__ which pulls magi_attention)
_tree = _load_module("verl.utils.prefix_tree.tree", os.path.join(_PKG_DIR, "tree.py"))
_sg = _load_module("verl.utils.prefix_tree.segment_grouper", os.path.join(_PKG_DIR, "segment_grouper.py"))

# dynamic.py imports magi_attention — stub it via conftest first, then load
# (conftest.py at this dir already stubs magi_attention/megatron/apex/transformer_engine)
# Re-run the stub registration in case test is run standalone.
import importlib  # noqa: E402

try:
    importlib.import_module("magi_attention")
except ModuleNotFoundError:
    _stub = types.ModuleType("magi_attention")
    _stub.__path__ = []
    sys.modules["magi_attention"] = _stub
    _api = types.ModuleType("magi_attention.api")
    _api.DistAttnConfig = type("DistAttnConfig", (), {})
    _api.get_position_ids = lambda *a, **k: None
    _api.magi_attn_flex_key = lambda *a, **k: None
    _api.undispatch = lambda *a, **k: None
    sys.modules["magi_attention.api"] = _api
    _stub.api = _api

# Load dynamic.py (provides greedy_build_tries) — needs the magi stub above
try:
    _dyn = _load_module("verl.utils.prefix_tree.dynamic", os.path.join(_PKG_DIR, "dynamic.py"))
except Exception:
    _dyn = None  # dynamic.py has heavy deps; skip fallback test if load fails


# ── Local replica of ray_trainer._build_global_trie (avoids heavy deps) ──────
# NOTE: production ray_trainer._build_global_trie passes list[list[int]] to both
# build_global_tree_from_segments and greedy_build_tries. However,
# build_global_tree_from_segments calls .tolist() on samples[i] (tree.py:340),
# which only works on tensors — not plain lists. This is a latent production bug
# (see report). Here we pass tensors to build_global_tree_from_segments and
# lists to greedy_build_tries so both paths exercise correctly.
def _build_global_trie_local(seqs_t, seg_hashes=None, seg_lengths=None):
    """Returns (trie, leaf_idx) — replica of ray_trainer._build_global_trie logic.

    Args:
        seqs_t: list of 1-D torch.LongTensor (one per sequence).
    """
    total_raw = sum(int(s.numel()) for s in seqs_t)
    trie = None
    if seg_hashes is not None and seg_lengths is not None:
        trie = _tree.build_global_tree_from_segments(seqs_t, seg_hashes, seg_lengths)
    if trie is None and _dyn is not None:
        seqs_list = [s.tolist() for s in seqs_t]
        tries, _ = _dyn.greedy_build_tries(seqs_list, max_tokens_per_tree=total_raw * 10)
        if tries and total_raw > 0:
            trie = tries[0]
    if trie is None:
        return None, None

    leaf_idx = np.full(len(seqs_t), -1, dtype=np.int64)
    for flat_idx, node in enumerate(trie.nodes):
        if not node.children:
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = flat_idx
    return trie, leaf_idx


# ── Minimal DataProto.reorder stand-in (tests the contract) ──────────────────
class _BatchStandIn:
    """Minimal stand-in for DataProto's non_tensor_batch propagation.

    Mirrors verl/protocol.py:967-977 reorder() semantics: numpy arrays in
    non_tensor_batch are indexed via `val[indices_np]`.
    """

    def __init__(self, non_tensor_batch: dict):
        self.non_tensor_batch = non_tensor_batch

    def reorder(self, indices: torch.Tensor):
        indices_np = indices.detach().numpy()
        reordered = {}
        for key, val in self.non_tensor_batch.items():
            if isinstance(val, list):
                reordered[key] = [val[i] for i in indices_np]
            else:
                reordered[key] = val[indices_np]
        self.non_tensor_batch = reordered

    def chunk(self, chunks: int):
        n = len(next(iter(self.non_tensor_batch.values())))
        per = n // chunks
        result = []
        for i in range(chunks):
            sub = {k: v[i * per : (i + 1) * per] for k, v in self.non_tensor_batch.items()}
            result.append(_BatchStandIn(sub))
        return result


# ── Test fixtures ────────────────────────────────────────────────────────────
def _make_seqs(n_prompts: int = 2, rollout_n: int = 2, prompt_len: int = 5, resp_len: int = 3):
    """Build GRPO-style seqs (as 1-D LongTensors) + segment metadata.

    Returns tensors because build_global_tree_from_segments calls .tolist() on
    samples[i] (tree.py:340) — plain lists would crash. The fallback path
    (greedy_build_tries) receives lists via conversion inside _build_global_trie_local.
    """
    seqs = []
    uids = []
    for p in range(n_prompts):
        prompt_tokens = list(range(100 + p * 10, 100 + p * 10 + prompt_len))
        for r in range(rollout_n):
            resp_tokens = list(range(200 + (p * rollout_n + r) * 10, 200 + (p * rollout_n + r) * 10 + resp_len))
            seqs.append(torch.tensor(prompt_tokens + resp_tokens, dtype=torch.long))
            uids.append(f"p{p}")
    prompt_lengths = [prompt_len] * len(seqs)
    seg_hashes, seg_lengths = _sg.create_grpo_segment_metadata(uids, prompt_lengths, rollout_n)
    return seqs, seg_hashes, seg_lengths, n_prompts, rollout_n


# ── Tests ────────────────────────────────────────────────────────────────────
def test_build_global_trie_attaches_trie_and_leaf_idx():
    """Building the trie should produce a non-empty trie and a valid leaf_idx array."""
    seqs, seg_hashes, seg_lengths, n_prompts, rollout_n = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs, seg_hashes, seg_lengths)

    assert trie is not None
    assert len(trie.nodes) > 0
    assert leaf_idx is not None
    assert isinstance(leaf_idx, np.ndarray)
    assert leaf_idx.dtype == np.int64
    assert leaf_idx.shape == (n_prompts * rollout_n,)
    assert (leaf_idx >= 0).all(), "every sample must map to a leaf flat_idx"


def test_leaf_idx_survives_reorder():
    """leaf_idx must propagate through numpy fancy-index reordering (DataProto.reorder contract)."""
    seqs, seg_hashes, seg_lengths, _, _ = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs, seg_hashes, seg_lengths)

    batch = _BatchStandIn({"leaf_idx": leaf_idx.copy()})
    perm = torch.randperm(len(seqs))
    batch.reorder(perm)

    np.testing.assert_array_equal(batch.non_tensor_batch["leaf_idx"], leaf_idx[perm.numpy()])


def test_leaf_idx_survives_chunk():
    """leaf_idx must propagate through chunking (DataProto.chunk contract)."""
    seqs, seg_hashes, seg_lengths, _, _ = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs, seg_hashes, seg_lengths)

    batch = _BatchStandIn({"leaf_idx": leaf_idx.copy()})
    chunks = batch.chunk(chunks=2)
    assert len(chunks) == 2
    concatenated = np.concatenate([c.non_tensor_batch["leaf_idx"] for c in chunks], axis=0)
    np.testing.assert_array_equal(concatenated, leaf_idx)


def test_leaf_idx_points_to_valid_leaf_nodes():
    """Each leaf_idx[i] must point to a leaf node (no children) in the trie."""
    seqs, seg_hashes, seg_lengths, _, _ = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs, seg_hashes, seg_lengths)

    for i, flat_idx in enumerate(leaf_idx):
        node = trie.nodes[flat_idx]
        assert not node.children, f"sample {i} maps to non-leaf node {flat_idx}"
        assert i in node.sequence_ids, f"sample {i} not in leaf {flat_idx}.sequence_ids={node.sequence_ids}"


def test_build_global_trie_without_segment_metadata_falls_back():
    """When segment_hashes/segment_lengths are absent, fall back to greedy_build_tries."""
    if _dyn is None:
        import pytest

        pytest.skip("dynamic.py could not be loaded — greedy_build_tries unavailable")

    seqs, _, _, _, _ = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs, seg_hashes=None, seg_lengths=None)

    assert trie is not None
    assert leaf_idx is not None
    assert (leaf_idx >= 0).all()


# ── Multilevel segment tests ────────────────────────────────────────────────
def _make_multilevel_seqs():
    """Build multi-turn GRPO seqs: 2 prompts × 2 rollouts, 2 shared turns each.

    Layout per sample: [turn1 (shared) | turn2 (shared) | response (unique)]
    Segments cover only the shared turns; response is the unshared suffix.
    """
    # p0: turn1=[10,11,12], turn2=[20,21]
    # p1: turn1=[50,51,52], turn2=[60,61]
    seqs = [
        torch.tensor([10, 11, 12, 20, 21, 30, 31], dtype=torch.long),  # p0 r0
        torch.tensor([10, 11, 12, 20, 21, 32, 33], dtype=torch.long),  # p0 r1
        torch.tensor([50, 51, 52, 60, 61, 70, 71], dtype=torch.long),  # p1 r0
        torch.tensor([50, 51, 52, 60, 61, 72, 73], dtype=torch.long),  # p1 r1
    ]
    # Two segments per sample: (turn1_hash, turn1_len=3), (turn2_hash, turn2_len=2)
    segments = [
        [("p0_t1", 3), ("p0_t2", 2)],
        [("p0_t1", 3), ("p0_t2", 2)],
        [("p1_t1", 3), ("p1_t2", 2)],
        [("p1_t1", 3), ("p1_t2", 2)],
    ]
    seg_hashes, seg_lengths = _sg.create_segment_metadata(segments)
    return seqs, seg_hashes, seg_lengths


def test_multilevel_segment_builds_intermediate_ancestor_nodes():
    """Multilevel segments must produce ancestor nodes at each shared level, not
    flatten everything beyond level 0 into a single leaf."""
    seqs, seg_hashes, seg_lengths = _make_multilevel_seqs()
    trie = _tree.build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)

    assert trie is not None
    # Expected: 2 prefix (turn1) + 2 intermediate (turn2) + 4 leaves = 8 nodes
    assert len(trie.nodes) == 8, f"expected 8 nodes, got {len(trie.nodes)}"

    # Classify nodes by depth (ancestor chain length)
    leaves = [n for n in trie.nodes if not n.children]
    intermediates = [n for n in trie.nodes if n.children and n.ancestor is not None]
    roots = [n for n in trie.nodes if n.ancestor is None]
    assert len(leaves) == 4, f"expected 4 leaves, got {len(leaves)}"
    assert len(intermediates) == 2, f"expected 2 intermediate nodes, got {len(intermediates)}"
    assert len(roots) == 2, f"expected 2 root nodes, got {len(roots)}"

    # Each intermediate's ancestor must be a root prefix node
    for inter in intermediates:
        assert inter.ancestor in roots, "intermediate must hang off a root prefix node"

    # Each leaf's ancestor must be an intermediate node (not a root)
    for leaf in leaves:
        assert leaf.ancestor in intermediates, "leaf must hang off an intermediate node"
        assert leaf.ancestor is not None, "leaf must have a non-None ancestor"


def test_multilevel_leaf_input_ids_are_response_only():
    """Leaf input_ids must contain only the unshared response, not turn1/turn2 tokens."""
    seqs, seg_hashes, seg_lengths = _make_multilevel_seqs()
    trie = _tree.build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)

    expected_responses = [[30, 31], [32, 33], [70, 71], [72, 73]]
    for sid, expected in enumerate(expected_responses):
        leaf = trie.leaves[sid]
        assert leaf is not None, f"no leaf for sample {sid}"
        assert leaf.input_ids == expected, (
            f"sample {sid} leaf input_ids={leaf.input_ids}, expected {expected}"
        )


def test_multilevel_ancestor_input_ids_are_segment_tokens():
    """Root prefix nodes carry turn1 tokens; intermediate nodes carry turn2 tokens."""
    seqs, seg_hashes, seg_lengths = _make_multilevel_seqs()
    trie = _tree.build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)

    roots = [n for n in trie.nodes if n.ancestor is None and n.children]
    root_token_sets = {tuple(n.input_ids) for n in roots}
    assert {(10, 11, 12), (50, 51, 52)} == root_token_sets, (
        f"root prefix tokens mismatch: {root_token_sets}"
    )

    intermediates = [n for n in trie.nodes if n.ancestor is not None and n.children]
    inter_token_sets = {tuple(n.input_ids) for n in intermediates}
    assert {(20, 21), (60, 61)} == inter_token_sets, (
        f"intermediate token mismatch: {inter_token_sets}"
    )


def test_multilevel_leaves_cover_all_samples():
    """trie.leaves[sid] must be populated for every sample."""
    seqs, seg_hashes, seg_lengths = _make_multilevel_seqs()
    trie = _tree.build_global_tree_from_segments(seqs, seg_hashes, seg_lengths)

    assert len(trie.leaves) == 4
    for sid in range(4):
        assert trie.leaves[sid] is not None, f"trie.leaves[{sid}] is None"
        assert sid in trie.leaves[sid].sequence_ids
