"""Tests for SSMPrefixCache — no model required.

Tests:
  1. Rolling hash: same token sequence → same hash; different → different
  2. Hash chaining: prefix_of(A) ≠ hash_of(B) even if last chunk matches
  3. lookup: returns deepest matching boundary
  4. insert + LRU eviction
  5. No cross-contamination: different prefixes hash differently
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ssm_prefix_cache import SSMPrefixCache, chunk_rolling_hashes, _EMPTY_HASH

CHUNK = 4   # small for testing

# ---------------------------------------------------------------------------
# 1. Hash consistency
# ---------------------------------------------------------------------------

def test_hash_determinism():
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    h1 = chunk_rolling_hashes(ids, CHUNK)
    h2 = chunk_rolling_hashes(ids, CHUNK)
    assert h1 == h2, "hashes must be deterministic"
    print("✓ hash determinism")


def test_hash_different_tokens():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    b = torch.tensor([1, 2, 3, 5], dtype=torch.long)   # differs in last token
    assert chunk_rolling_hashes(a, CHUNK) != chunk_rolling_hashes(b, CHUNK)
    print("✓ different tokens → different hash")


def test_hash_chaining():
    """Two sequences sharing chunk_0 but differing in chunk_1 must produce
    different h[1] even though h[0] is identical."""
    base    = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    seq_a   = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    seq_b   = torch.tensor([1, 2, 3, 4, 5, 6, 7, 9], dtype=torch.long)

    h_base  = chunk_rolling_hashes(base,  CHUNK)
    h_a     = chunk_rolling_hashes(seq_a, CHUNK)
    h_b     = chunk_rolling_hashes(seq_b, CHUNK)

    # chunk_0 must match between all three
    assert h_base[0] == h_a[0] == h_b[0], "shared first chunk → same h[0]"
    # chunk_1 must differ between a and b
    assert h_a[1] != h_b[1], "different chunk_1 → different h[1]"
    print("✓ hash chaining correct")


def test_incomplete_chunk_ignored():
    """Tokens not filling a complete chunk produce no hash entry."""
    ids = torch.tensor([1, 2, 3], dtype=torch.long)   # 3 < CHUNK=4
    assert chunk_rolling_hashes(ids, CHUNK) == [], "incomplete chunk → no hash"
    print("✓ incomplete chunk ignored")


# ---------------------------------------------------------------------------
# 2. Cache lookup
# ---------------------------------------------------------------------------

class FakeSnap:
    def __init__(self, label):
        self.label = label
        self.cleared = False
    def clear(self):
        self.cleared = True
    def __repr__(self):
        return f"Snap({self.label})"


def test_lookup_miss():
    cache = SSMPrefixCache(capacity=16, chunk_size=CHUNK)
    ids = torch.arange(8, dtype=torch.long)
    pos, snap = cache.lookup(ids)
    assert pos == 0 and snap is None
    print("✓ lookup miss → (0, None)")


def test_lookup_hit():
    cache = SSMPrefixCache(capacity=16, chunk_size=CHUNK)
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    snaps = [FakeSnap("chunk0"), FakeSnap("chunk1")]
    cache.insert(ids, snaps)

    # Lookup with exactly the same prefix
    pos, snap = cache.lookup(ids)
    assert pos == 8, f"expected 8, got {pos}"
    assert snap.label == "chunk1"
    print("✓ lookup exact hit → deepest boundary")


def test_lookup_partial_match():
    """Cache has [1..4]. Query is [1..4, 5..8] → should hit at pos=4."""
    cache = SSMPrefixCache(capacity=16, chunk_size=CHUNK)
    short = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    cache.insert(short, [FakeSnap("c0")])

    long_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    pos, snap = cache.lookup(long_ids)
    assert pos == 4, f"expected 4, got {pos}"
    assert snap.label == "c0"
    print("✓ lookup partial match → pos=4")


def test_no_cross_contamination():
    """Two sequences with same final chunk but different leading chunks
    should NOT share a cache hit beyond their actual common prefix."""
    cache = SSMPrefixCache(capacity=16, chunk_size=CHUNK)

    # seq_a: [1,2,3,4, 5,6,7,8]
    seq_a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    snaps_a = [FakeSnap("a0"), FakeSnap("a1")]
    cache.insert(seq_a, snaps_a)

    # seq_b: [9,10,11,12, 5,6,7,8] — chunk_1 tokens same but chain differs
    seq_b = torch.tensor([9, 10, 11, 12, 5, 6, 7, 8], dtype=torch.long)
    pos, snap = cache.lookup(seq_b)
    assert pos == 0, f"different prefix → no hit, got pos={pos}"
    print("✓ no cross-contamination between different prefix chains")


# ---------------------------------------------------------------------------
# 3. LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction():
    cache = SSMPrefixCache(capacity=2, chunk_size=CHUNK)

    seq_a = torch.tensor([1, 2, 3, 4], dtype=torch.long)   # chunk 0: key=h_a
    seq_b = torch.tensor([5, 6, 7, 8], dtype=torch.long)   # chunk 0: key=h_b
    seq_c = torch.tensor([9, 10, 11, 12], dtype=torch.long) # chunk 0: key=h_c

    snap_a = FakeSnap("a")
    snap_b = FakeSnap("b")
    snap_c = FakeSnap("c")

    cache.insert(seq_a, [snap_a])   # size=1
    cache.insert(seq_b, [snap_b])   # size=2  (full)
    cache.insert(seq_c, [snap_c])   # size=3 → evict LRU (a)

    assert len(cache) == 2
    assert snap_a.cleared, "LRU entry (a) should have been cleared"

    # b and c should still be present
    pos_b, s_b = cache.lookup(seq_b)
    pos_c, s_c = cache.lookup(seq_c)
    assert pos_b == 4 and s_b.label == "b"
    assert pos_c == 4 and s_c.label == "c"
    print("✓ LRU eviction: oldest entry evicted, .clear() called")


def test_lru_access_refreshes():
    """Access seq_a after seq_b is inserted; then insert seq_c → b evicted, not a."""
    cache = SSMPrefixCache(capacity=2, chunk_size=CHUNK)

    seq_a = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    seq_b = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    seq_c = torch.tensor([9, 10, 11, 12], dtype=torch.long)

    cache.insert(seq_a, [FakeSnap("a")])
    cache.insert(seq_b, [FakeSnap("b")])

    # Touch a (makes b the LRU)
    cache.lookup(seq_a)

    snap_c = FakeSnap("c")
    cache.insert(seq_c, [snap_c])   # evicts b (now LRU)

    pos_a, s_a = cache.lookup(seq_a)
    pos_b, s_b = cache.lookup(seq_b)
    assert pos_a == 4 and s_a is not None, "a should survive"
    assert pos_b == 0 and s_b is None, "b should be evicted"
    print("✓ LRU access refresh: lookup promotes entry, correct eviction")


# ---------------------------------------------------------------------------
# 4. Linear attention key property
# ---------------------------------------------------------------------------

def test_last_chunk_hash_as_key():
    """For linear attention, only the LAST chunk's hash matters as lookup key.
    Verify that inserting a long sequence lets us hit at the deepest boundary
    even when the query sequence is longer (but shares the same prefix chain).
    """
    cache = SSMPrefixCache(capacity=16, chunk_size=CHUNK)

    # Insert 3 chunks
    seq = torch.tensor([1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12], dtype=torch.long)
    snaps = [FakeSnap(f"s{i}") for i in range(3)]
    cache.insert(seq, snaps)

    # Lookup with 4 chunks (same prefix + new chunk 3)
    longer = torch.tensor([1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16], dtype=torch.long)
    pos, snap = cache.lookup(longer)
    # Should hit at pos=12 (last of the 3 cached chunks)
    assert pos == 12, f"expected 12, got {pos}"
    assert snap.label == "s2"
    print("✓ last chunk hash: deepest cached boundary found correctly")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_hash_determinism()
    test_hash_different_tokens()
    test_hash_chaining()
    test_incomplete_chunk_ignored()
    test_lookup_miss()
    test_lookup_hit()
    test_lookup_partial_match()
    test_no_cross_contamination()
    test_lru_eviction()
    test_lru_access_refreshes()
    test_last_chunk_hash_as_key()
    print("\nAll cache tests passed.")


# ---------------------------------------------------------------------------
# Integration test: real model
# ---------------------------------------------------------------------------

def test_with_model():
    """Run Qwen3.5-0.8B, cache states at chunk boundaries, verify lookup
    gives identical output to a fresh full run."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "verl-prefix-tree"))

    import transformers.models.qwen3_5.modeling_qwen3_5 as _m
    _m.causal_conv1d_fn = None
    _m.causal_conv1d_update = None
    _m.chunk_gated_delta_rule = None
    _m.fused_recurrent_gated_delta_rule = None
    _m.FusedRMSNormGated = None

    from transformers import AutoModelForCausalLM
    from ssm_branch_cache import (
        _NodeSnapshot, apply_ssm_patch, _make_dyn_cache,
        _run_segment, _run_tokens_one_by_one,
    )

    MODEL_BASE = os.environ.get(
        "MODEL_BASE",
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/Qwen",
    )
    model = AutoModelForCausalLM.from_pretrained(
        f"{MODEL_BASE}/Qwen3.5-0.8B", dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    apply_ssm_patch()

    CHUNK = 4
    cache = SSMPrefixCache(capacity=64, chunk_size=CHUNK)

    # Full sequence: 3 complete chunks + decode token
    prefix = torch.tensor([[1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]], dtype=torch.long)
    decode_tok = torch.tensor([[99]], dtype=torch.long)

    # --- Reference: run full prefix then decode ---
    dyn_ref = _make_dyn_cache(model)
    _run_segment(model, prefix, dyn_ref)
    h_ref = _run_tokens_one_by_one(model, decode_tok, dyn_ref)  # (1,1,H)

    # --- Build cache: insert snapshots at each chunk boundary ---
    dyn_build = _make_dyn_cache(model)
    snaps = []
    for c in range(prefix.shape[1] // CHUNK):
        chunk_tok = prefix[:, c*CHUNK:(c+1)*CHUNK]
        if c == 0:
            _run_segment(model, chunk_tok, dyn_build)
        else:
            _run_tokens_one_by_one(model, chunk_tok, dyn_build)
        snaps.append(_NodeSnapshot.from_dyn_cache(dyn_build))
    cache.insert(prefix.flatten(), snaps)

    # --- Lookup + resume ---
    hit_pos, cached_snap = cache.lookup(prefix.flatten())
    assert hit_pos == 12, f"expected hit at 12, got {hit_pos}"

    dyn_cached = _make_dyn_cache(model)
    cached_snap.restore_to(dyn_cached)
    h_cached = _run_tokens_one_by_one(model, decode_tok, dyn_cached)  # (1,1,H)

    diff = (h_ref - h_cached).abs().max().item()
    ok = diff < 1e-4
    print(f"  {'✓' if ok else '✗'} model integration: max_diff={diff:.2e}  hit_pos={hit_pos}")
    assert ok, f"cache restore mismatch: diff={diff}"


if __name__ == "__main__":
    test_hash_determinism()
    test_hash_different_tokens()
    test_hash_chaining()
    test_incomplete_chunk_ignored()
    test_lookup_miss()
    test_lookup_hit()
    test_lookup_partial_match()
    test_no_cross_contamination()
    test_lru_eviction()
    test_lru_access_refreshes()
    test_last_chunk_hash_as_key()
    print("\nAll cache unit tests passed.")
    print("\nRunning model integration test...")
    test_with_model()
    print("All tests passed.")
