#!/usr/bin/env python3
# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
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
"""Mask correctness test: 2 micro-batches, each with 10 groups × 8 rollouts.

Build via real greedy_build_tries + subtrie_view, check the dense attention
mask has zero cross-group entries.
"""

import torch
from verl.utils.prefix_tree.dynamic import greedy_build_tries, subtrie_view
from verl.utils.prefix_tree.utils import build_layout_from_tree_node, build_prefix_tree_dense_mask


# ── Sequence generation ───────────────────────────────────────────────────────

N_GROUPS   = 10
N_ROLLOUTS = 8
PREFIX_LEN = 5
RESP_LEN   = 3


def make_mbs_sequences(mbs_id: int) -> list[list[int]]:
    """10 groups × 8 rollouts, each group sharing a 5-token prefix.

    Token space: mbs_id*100_000 + group*1000 + position — no overlap across
    groups or micro-batches.
    """
    seqs = []
    base = mbs_id * 100_000
    for g in range(N_GROUPS):
        prefix = [base + g * 1000 + i for i in range(PREFIX_LEN)]
        for r in range(N_ROLLOUTS):
            resp = [base + g * 1000 + 500 + r * 10 + j for j in range(RESP_LEN)]
            seqs.append(prefix + resp)
    return seqs  # 80 sequences, indices 0..79; group g owns seqs[g*8 .. g*8+7]


def build_subtrie(seqs: list[list[int]]):
    total = sum(len(s) for s in seqs)
    tries, _ = greedy_build_tries(seqs, max_tokens_per_tree=total * 10)
    assert len(tries) == 1, f"expected single forest, got {len(tries)}"
    trie = tries[0]
    all_ids = set(range(len(seqs)))
    return subtrie_view(trie, all_ids)


# ── Mask verification ─────────────────────────────────────────────────────────

def verify_mask(params, seqs: list[list[int]], mbs_id: int):
    """Assert block-shape correctness of the dense attention mask."""
    total = params.total_seqlen_q
    mask = build_prefix_tree_dense_mask(
        total, params.q_ranges, params.k_ranges, params.mask_types
    )
    q_pos, k_pos = mask.nonzero(as_tuple=True)
    ok = True

    # ── 1. Coverage: every flat token appears in at least one q_range ─────
    covered = torch.zeros(total, dtype=torch.bool)
    for qs, qe in params.q_ranges:
        covered[qs:qe] = True
    missing = (~covered).nonzero(as_tuple=True)[0]
    if missing.numel():
        print(f"  FAIL mbs={mbs_id}: {missing.numel()} uncovered flat positions (no q_range): "
              f"{missing[:5].tolist()}")
        ok = False
    else:
        print(f"  PASS mbs={mbs_id}: coverage — all {total} flat tokens in some q_range")

    # ── 2. Cross-group isolation ───────────────────────────────────────────
    token_group = torch.full((total,), -1, dtype=torch.long)
    for leaf_idx, sample_idx in enumerate(params.leaf_to_sample):
        g = sample_idx // N_ROLLOUTS
        for a, b in params._leaf_ancestor_ranges[leaf_idx]:
            token_group[a:b] = g
        ls, le = params.leaf_ranges[leaf_idx]
        token_group[ls:le] = g

    cross = [(q.item(), k.item()) for q, k in zip(q_pos, k_pos)
             if token_group[q] >= 0 and token_group[k] >= 0
             and token_group[q] != token_group[k]]
    if cross:
        print(f"  FAIL mbs={mbs_id}: {len(cross)} cross-group attention pairs")
        for q, k in cross[:3]:
            print(f"    pos {q} (g{token_group[q].item()}) → pos {k} (g{token_group[k].item()})")
        ok = False
    else:
        print(f"  PASS mbs={mbs_id}: isolation — 0 cross-group pairs in {q_pos.shape[0]} total")

    # ── 3. Causal within each segment (prefix block and each leaf block) ──
    # Mark each flat position with its segment id (prefix=0, leaf_i=i+1 per group).
    token_seg = torch.full((total,), -1, dtype=torch.long)
    seg_id = 0
    seen_anc: set[tuple] = set()
    for leaf_idx in range(len(params.leaf_to_sample)):
        for a, b in params._leaf_ancestor_ranges[leaf_idx]:
            if (a, b) not in seen_anc and b > a:
                token_seg[a:b] = seg_id
                seg_id += 1
                seen_anc.add((a, b))
        ls, le = params.leaf_ranges[leaf_idx]
        if le > ls:
            token_seg[ls:le] = seg_id
            seg_id += 1

    acausal = [(q.item(), k.item()) for q, k in zip(q_pos, k_pos)
               if token_seg[q] >= 0 and token_seg[k] >= 0
               and token_seg[q] == token_seg[k]   # same segment
               and k > q]                          # k is AFTER q → anti-causal
    if acausal:
        print(f"  FAIL mbs={mbs_id}: {len(acausal)} anti-causal pairs within same segment")
        for q, k in acausal[:3]:
            print(f"    mask[{q},{k}]=1 but k>q in same segment {token_seg[q].item()}")
        ok = False
    else:
        print(f"  PASS mbs={mbs_id}: causal — no anti-causal within-segment pairs")

    # ── 4. Full leaf→prefix completeness ──────────────────────────────────
    fail_full = 0
    for leaf_idx, sample_idx in enumerate(params.leaf_to_sample):
        ls, le = params.leaf_ranges[leaf_idx]
        if le <= ls:
            continue
        for a, b in params._leaf_ancestor_ranges[leaf_idx]:
            if b <= a:
                continue
            # Every (q in leaf, k in ancestor) should be attended
            leaf_t = torch.arange(ls, le)
            anc_t  = torch.arange(a, b)
            block = mask[leaf_t.unsqueeze(1), anc_t.unsqueeze(0)]  # (leaf_len, anc_len)
            if not block.all():
                zeros = (~block).sum().item()
                fail_full += zeros
    if fail_full:
        print(f"  FAIL mbs={mbs_id}: {fail_full} missing leaf→ancestor FULL entries")
        ok = False
    else:
        print(f"  PASS mbs={mbs_id}: full — leaf→ancestor blocks are all-ones")

    return ok


def verify_reconstruction(params, seqs: list[list[int]], mbs_id: int) -> bool:
    """Verify expand round-trip: cat(ancestors + leaf) == original sequence."""
    flat = params.tree_packed_tokens
    n = len(seqs)
    result = [None] * n
    for leaf_idx, sample_idx in enumerate(params.leaf_to_sample):
        anc = params._leaf_ancestor_ranges[leaf_idx]
        ls, le = params.leaf_ranges[leaf_idx]
        parts = [flat[a:b] for a, b in anc] + [flat[ls:le]]
        result[sample_idx] = torch.cat(parts)

    for i, (orig_seq, recon) in enumerate(zip(seqs, result)):
        orig = torch.tensor(orig_seq)
        if not torch.equal(orig, recon):
            print(f"  FAIL mbs={mbs_id} sample {i}: {orig.tolist()} != {recon.tolist()}")
            return False

    print(f"  PASS mbs={mbs_id}: reconstruction correct for all {n} samples")
    return True


def verify_boundary_labels(params, seqs: list[list[int]], mbs_id: int) -> bool:
    """Label at flat position prefix_len-1 in each leaf = that leaf's first response token."""
    flat_labels = params.tree_packed_labels
    all_ok = True
    for leaf_idx, sample_idx in enumerate(params.leaf_to_sample):
        anc = params._leaf_ancestor_ranges[leaf_idx]
        ls, le = params.leaf_ranges[leaf_idx]
        parts_labels = [flat_labels[a:b] for a, b in anc] + [flat_labels[ls:le]]
        recon_labels = torch.cat(parts_labels)

        # Position PREFIX_LEN-1 is the last prefix position; its label = first response token
        expected = seqs[sample_idx][PREFIX_LEN]   # first token after prefix
        actual = recon_labels[PREFIX_LEN - 1].item()
        if actual != expected:
            print(f"  FAIL mbs={mbs_id} sample {sample_idx}: boundary label={actual}, want={expected}")
            all_ok = False

    if all_ok:
        print(f"  PASS mbs={mbs_id}: boundary labels correct for all {len(seqs)} samples")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def run_mbs(mbs_id: int) -> bool:
    print(f"\n=== mbs {mbs_id} ({N_GROUPS} groups × {N_ROLLOUTS} rollouts) ===")
    seqs = make_mbs_sequences(mbs_id)
    samples = [torch.tensor(s, dtype=torch.long) for s in seqs]
    subtrie = build_subtrie(seqs)
    assert subtrie is not None, "subtrie_view returned None"

    params = build_layout_from_tree_node(samples, subtrie)

    raw_tokens = sum(len(s) for s in seqs)
    flat_tokens = params.total_seqlen_q
    print(f"  raw={raw_tokens}  flat={flat_tokens}  "
          f"dedup_ratio={1 - flat_tokens/raw_tokens:.3f}")

    ok = True
    ok &= verify_reconstruction(params, seqs, mbs_id)
    ok &= verify_boundary_labels(params, seqs, mbs_id)
    ok &= verify_mask(params, seqs, mbs_id)
    return ok


def build_reference_mask(params) -> torch.Tensor:
    """Ground-truth mask from first principles — no tree code involved.

    Rules:
      - Causal within every prefix segment (CAUSAL rect)
      - Causal within every leaf segment (CAUSAL rect)
      - Full from every leaf segment to every ancestor segment (FULL rect)
    """
    total = params.total_seqlen_q
    ref = torch.zeros(total, total, dtype=torch.bool)

    seen_anc: set[tuple] = set()
    prefix_segs: list[tuple[int, int]] = []
    leaf_anc_map: list[list[tuple[int, int]]] = []  # per leaf: its ancestor ranges

    for leaf_idx in range(len(params.leaf_to_sample)):
        ancs = []
        for a, b in params._leaf_ancestor_ranges[leaf_idx]:
            if b > a:
                if (a, b) not in seen_anc:
                    seen_anc.add((a, b))
                    prefix_segs.append((a, b))
                    # causal within prefix segment
                    for q in range(a, b):
                        ref[q, a : q + 1] = True
                ancs.append((a, b))
        ls, le = params.leaf_ranges[leaf_idx]
        if le > ls:
            # causal within leaf
            for q in range(ls, le):
                ref[q, ls : q + 1] = True
            # full from leaf to each ancestor
            for a, b in ancs:
                ref[ls:le, a:b] = True
        leaf_anc_map.append(ancs)

    return ref


def verify_against_reference(params, mbs_id: int) -> bool:
    """Compare tree-built mask against the reference mask."""
    total = params.total_seqlen_q
    actual = build_prefix_tree_dense_mask(
        total, params.q_ranges, params.k_ranges, params.mask_types
    )
    ref = build_reference_mask(params)
    if torch.equal(actual, ref):
        print(f"  PASS mbs={mbs_id}: mask == reference (exact match, {actual.sum().item()} pairs)")
        return True
    extra  = (actual & ~ref).sum().item()
    missing = (~actual & ref).sum().item()
    print(f"  FAIL mbs={mbs_id}: mask differs — {extra} extra, {missing} missing attention pairs")
    return False


def run_mutation_tests(params) -> bool:
    """Inject each violation type; confirm every check fires."""
    total = params.total_seqlen_q
    ref = build_reference_mask(params)
    ok = True

    def _check_fires(name, bad_mask, check_fn):
        nonlocal ok
        fired = not check_fn(bad_mask)
        status = "PASS" if fired else "FAIL"
        if not fired:
            ok = False
        print(f"  {status} mutation/{name}: check {'fired' if fired else 'MISSED violation'}")

    # pick a known prefix seg and leaf seg from group 0
    leaf0_idx = 0
    ls0, le0 = params.leaf_ranges[leaf0_idx]
    anc0 = [(a, b) for a, b in params._leaf_ancestor_ranges[leaf0_idx] if b > a]
    pa0, pb0 = anc0[0] if anc0 else (0, 1)

    # pick a leaf from group 1 (first rollout of second group)
    leaf1_idx = N_ROLLOUTS
    ls1, le1 = params.leaf_ranges[leaf1_idx]

    # M1: coverage — reference diff catches it (if a position has no q_range, it
    # can't appear in the actual mask, so actual≠ref). Tested implicitly above.

    # M2: inject cross-group attention
    def check_no_cross(m):
        token_group = torch.full((total,), -1, dtype=torch.long)
        for li, si in enumerate(params.leaf_to_sample):
            g = si // N_ROLLOUTS
            for a, b in params._leaf_ancestor_ranges[li]:
                token_group[a:b] = g
            lls, lle = params.leaf_ranges[li]
            token_group[lls:lle] = g
        q_pos, k_pos = m.nonzero(as_tuple=True)
        return not any(
            token_group[q] >= 0 and token_group[k] >= 0 and token_group[q] != token_group[k]
            for q, k in zip(q_pos.tolist(), k_pos.tolist())
        )
    bad = ref.clone()
    if ls0 < le0 and ls1 < le1:
        bad[ls0, ls1] = True   # leaf-0 attends leaf-1 (cross-group)
    _check_fires("cross_group", bad, check_no_cross)

    # M3: inject anti-causal within prefix
    def check_causal(m):
        token_seg = torch.full((total,), -1, dtype=torch.long)
        seg_id = 0
        seen: set = set()
        for li in range(len(params.leaf_to_sample)):
            for a, b in params._leaf_ancestor_ranges[li]:
                if (a, b) not in seen and b > a:
                    token_seg[a:b] = seg_id; seg_id += 1; seen.add((a, b))
            lls, lle = params.leaf_ranges[li]
            if lle > lls:
                token_seg[lls:lle] = seg_id; seg_id += 1
        q_pos, k_pos = m.nonzero(as_tuple=True)
        return not any(
            token_seg[q] >= 0 and token_seg[k] >= 0
            and token_seg[q] == token_seg[k] and k > q
            for q, k in zip(q_pos.tolist(), k_pos.tolist())
        )
    bad = ref.clone()
    if pb0 - pa0 >= 2:
        bad[pa0, pa0 + 1] = True   # token pa0 attends pa0+1 (future, anti-causal)
    _check_fires("anti_causal", bad, check_causal)

    # M4: remove an entry from a leaf→prefix full block
    def check_full(m):
        for li in range(len(params.leaf_to_sample)):
            lls, lle = params.leaf_ranges[li]
            if lle <= lls:
                continue
            for a, b in params._leaf_ancestor_ranges[li]:
                if b <= a:
                    continue
                if not m[lls:lle, a:b].all():
                    return False
        return True
    bad = ref.clone()
    if ls0 < le0 and pb0 > pa0:
        bad[ls0, pa0] = False   # remove one leaf→prefix entry
    _check_fires("missing_full_entry", bad, check_full)

    return ok


if __name__ == "__main__":
    print("=== functional tests ===")
    passed = all(run_mbs(mbs_id) for mbs_id in range(2))

    print("\n=== reference cross-validation ===")
    seqs = make_mbs_sequences(0)
    samples = [torch.tensor(s, dtype=torch.long) for s in seqs]
    subtrie = build_subtrie(seqs)
    params = build_layout_from_tree_node(samples, subtrie)
    passed &= verify_against_reference(params, mbs_id=0)

    print("\n=== mutation tests (each check must fire on bad input) ===")
    passed &= run_mutation_tests(params)

    print("\n" + ("ALL PASSED" if passed else "SOME FAILED"))
