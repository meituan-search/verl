"""Correctness test for ssm_branch_cache.py.

Tests the branch-cache on Qwen3.5-0.8B CPU (torch fallbacks enabled).

Tree design:
    root = [1,2,3,4,5]   — shared prompt prefix (prefill / chunk mode)
    ├── leaf_a = [8]      — single-token decode from root state
    └── leaf_b = [9]      — single-token decode from root state

    root_child_a = [1,2,3,4,5,6]  — deeper tree
    ├── leaf_c = [8]
    └── leaf_d = [9]

Naive reference:
    prefill([1,2,3,4,5]) → state S5
    decode([8]) from S5 → leaf_a output
    decode([9]) from S5 → leaf_b output

Both should match exactly because the single-token decode path uses the
exact recurrent_gated_delta_rule (not the approximate chunk variant).

Run: python3 test_ssm_branch_cache.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl-prefix-tree"))

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from ssm_branch_cache import (
    SSMBranchCache, _NodeSnapshot, apply_ssm_patch,
    ssm_tree_forward, _text_model, _run_segment, _run_tokens_one_by_one,
    _make_dyn_cache,
)

MODEL_BASE = os.environ.get(
    "MODEL_BASE",
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen",
)
MODEL_PATH = os.environ.get("MODEL_PATH", f"{MODEL_BASE}/Qwen3.5-0.8B")
DEVICE = "cpu"
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Force pure-torch fallbacks — causal_conv1d + FusedRMSNormGated are CUDA-only
# ---------------------------------------------------------------------------
import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35_mod
_qwen35_mod.causal_conv1d_fn = None
_qwen35_mod.causal_conv1d_update = None
_qwen35_mod.chunk_gated_delta_rule = None
_qwen35_mod.fused_recurrent_gated_delta_rule = None
_qwen35_mod.FusedRMSNormGated = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_PATH} on {DEVICE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Naive reference: prefill → decode
# ---------------------------------------------------------------------------

def naive_prefill_then_decode(model, prefix_ids, new_token_id) -> torch.Tensor:
    """Prefill prefix_ids then decode one new_token. Return last hidden state (1,H)."""
    tm = _text_model(model)
    dyn = _make_dyn_cache(model)

    # Prefill
    _run_segment(model, prefix_ids.unsqueeze(0).to(DEVICE), dyn, snap=None)

    # Decode one token
    tok = new_token_id.view(1, 1).to(DEVICE)
    out = _run_tokens_one_by_one(model, tok, dyn)   # (1, 1, H)
    return out[:, 0, :]   # (1, H)


# ---------------------------------------------------------------------------
# Test 1: snapshot / restore roundtrip
# ---------------------------------------------------------------------------

def test_snapshot_restore_roundtrip(model):
    print("\n[1] Snapshot / restore roundtrip ...")
    from ssm_branch_cache import _NodeSnapshot

    dyn1 = _make_dyn_cache(model)
    prefix = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(DEVICE)
    _run_segment(model, prefix, dyn1, snap=None)

    snap = _NodeSnapshot.from_dyn_cache(dyn1)

    dyn2 = _make_dyn_cache(model)
    snap.restore_to(dyn2)

    tok = torch.tensor([[8]], dtype=torch.long).to(DEVICE)
    h1 = _run_tokens_one_by_one(model, tok, dyn1)
    h2 = _run_tokens_one_by_one(model, tok, dyn2)

    diff = (h1 - h2).abs().max().item()
    ok = diff < ATOL
    print(f"  {'✓' if ok else '✗'} max_diff={diff:.2e}")
    assert ok, f"roundtrip failed: diff={diff}"


# ---------------------------------------------------------------------------
# Test 2: branch-cache matches naive for depth-1 tree
# ---------------------------------------------------------------------------

def test_depth1_branch(model):
    """Root [1,2,3,4,5], two single-token leaves [8] and [9]."""
    print("\n[2] Depth-1 tree (root=[1..5], leaves=[8],[9]) ...")
    from verl.utils.trajectory_tree import TrajectoryNode, TrajectoryTree

    root   = TrajectoryNode(tokens=torch.tensor([1,2,3,4,5], dtype=torch.long))
    leaf_a = TrajectoryNode(tokens=torch.tensor([8], dtype=torch.long), parent=root)
    leaf_b = TrajectoryNode(tokens=torch.tensor([9], dtype=torch.long), parent=root)
    root.children = [leaf_a, leaf_b]
    tree = TrajectoryTree(root=root)

    # Branch-cache
    cached = ssm_tree_forward(model, tree, device=DEVICE)   # [h_a, h_b]

    # Naive reference
    prefix = torch.tensor([1,2,3,4,5], dtype=torch.long)
    naive_a = naive_prefill_then_decode(model, prefix, torch.tensor(8))
    naive_b = naive_prefill_then_decode(model, prefix, torch.tensor(9))

    assert len(cached) == 2
    for name, cached_h, naive_h in [("leaf_a", cached[0], naive_a), ("leaf_b", cached[1], naive_b)]:
        diff = (cached_h - naive_h).abs().max().item()
        ok = diff < ATOL
        print(f"  {'✓' if ok else '✗'} {name}: max_diff={diff:.2e}")
        assert ok, f"{name} failed: diff={diff}"


# ---------------------------------------------------------------------------
# Test 3: depth-2 tree, all single-token branches
# ---------------------------------------------------------------------------

def test_depth2_branch(model):
    """Root [1,2,3], internal [6], [7], leaf each appends [8] or [9]."""
    print("\n[3] Depth-2 tree ...")
    from verl.utils.trajectory_tree import TrajectoryNode, TrajectoryTree

    root    = TrajectoryNode(tokens=torch.tensor([1,2,3], dtype=torch.long))
    child_a = TrajectoryNode(tokens=torch.tensor([6], dtype=torch.long), parent=root)
    child_b = TrajectoryNode(tokens=torch.tensor([7], dtype=torch.long), parent=root)
    leaf_aa = TrajectoryNode(tokens=torch.tensor([8], dtype=torch.long), parent=child_a)
    leaf_ab = TrajectoryNode(tokens=torch.tensor([9], dtype=torch.long), parent=child_a)
    leaf_ba = TrajectoryNode(tokens=torch.tensor([8], dtype=torch.long), parent=child_b)
    leaf_bb = TrajectoryNode(tokens=torch.tensor([9], dtype=torch.long), parent=child_b)

    root.children    = [child_a, child_b]
    child_a.children = [leaf_aa, leaf_ab]
    child_b.children = [leaf_ba, leaf_bb]
    tree = TrajectoryTree(root=root)

    # Branch-cache
    cached = ssm_tree_forward(model, tree, device=DEVICE)

    # Naive: chunk-prefill root [1,2,3] then decode each suffix token
    ROOT_LEN = 3   # length of root tokens in the tree
    def full_decode(tokens):
        dyn = _make_dyn_cache(model)
        # Chunk prefill for root portion
        root_toks = torch.tensor([tokens[:ROOT_LEN]], dtype=torch.long).to(DEVICE)
        _run_segment(model, root_toks, dyn, snap=None)
        # Decode suffix one by one
        for t in tokens[ROOT_LEN:]:
            out = _run_tokens_one_by_one(model, torch.tensor([[t]]).to(DEVICE), dyn)
        return out[:, 0, :]

    paths = [
        [1,2,3,6,8], [1,2,3,6,9], [1,2,3,7,8], [1,2,3,7,9]
    ]
    naive = [full_decode(p) for p in paths]

    assert len(cached) == 4
    names = ["leaf_aa", "leaf_ab", "leaf_ba", "leaf_bb"]
    all_ok = True
    for name, c, n in zip(names, cached, naive):
        diff = (c - n).abs().max().item()
        ok = diff < ATOL
        print(f"  {'✓' if ok else '✗'} {name}: max_diff={diff:.2e}")
        if not ok:
            all_ok = False
    assert all_ok, "depth-2 test failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = load_model()
    apply_ssm_patch()

    test_snapshot_restore_roundtrip(model)
    test_depth1_branch(model)
    test_depth2_branch(model)

    print("\nAll tests passed. SSM branch cache correct.")
