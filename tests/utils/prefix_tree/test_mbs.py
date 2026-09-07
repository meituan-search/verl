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


"""Unit tests for the reorder-safe micro-batch grouping API (grouping, balance
pass, PP/VPP divisibility of the micro-batch count, and DP equalization)."""

from __future__ import annotations

import socket
import statistics

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils import tensordict_utils as tu
from verl.utils.prefix_tree import dynamic as pt_dynamic
from verl.utils.prefix_tree.dynamic import (
    _mbs_groups_dfs,
    greedy_build_tries,
    mbs_groups_from_leaf_idx,
    prepare_prefix_tree_micro_batches,
    trie_group_flat_tokens,
)
from verl.utils.seqlen_balancing import roundup_divisible


def _make_samples(n_prompts, rollout_n, prefix_len, resp_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for p in range(n_prompts):
        prefix = torch.randint(0, 151936, (prefix_len,), generator=g)
        for _ in range(rollout_n):
            resp = torch.randint(0, 151936, (resp_len,), generator=g)
            samples.append(torch.cat([prefix, resp]))
    return samples


def _build_trie(samples):
    seq_lists = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]
    trie, _ = greedy_build_tries(seq_lists)
    return trie


def _leaf_idx_from_trie(trie, n_samples):
    """Build canonical leaf_idx: sample i -> its leaf's node_idx."""
    leaf_idx = torch.full((n_samples,), -1, dtype=torch.long)
    for node in trie.nodes:
        if not node.children:  # leaf
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node.node_idx
    assert int(leaf_idx.min().item()) >= 0, "trie has samples with no leaf"
    return leaf_idx


def test_mbs_groups_from_leaf_idx_covers_all_and_respects_budget():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    budget = 500
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=budget)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    for mb in mbs:
        assert trie_group_flat_tokens(mb, trie) <= budget


def test_mbs_groups_from_leaf_idx_reorder_safe():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx0 = _leaf_idx_from_trie(trie, len(samples))
    budget = 500
    perm = torch.randperm(len(samples), generator=torch.Generator().manual_seed(7)).tolist()
    leaf_idx1 = leaf_idx0[perm].clone()
    mbs = mbs_groups_from_leaf_idx(leaf_idx1, trie, max_token_len=budget)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    canon = mbs_groups_from_leaf_idx(leaf_idx0, trie, max_token_len=budget)
    canon_leaves = sorted(sorted({int(leaf_idx0[i]) for i in mb}) for mb in canon)
    perm_leaves = sorted(sorted({int(leaf_idx1[i]) for i in mb}) for mb in mbs)
    assert canon_leaves == perm_leaves


def test_mbs_groups_from_leaf_idx_duplicates_stay_together():
    base = [1, 2, 3, 4, 5]
    seqs = [base, base, [1, 2, 9, 9, 9], [1, 2, 7, 7, 7]]
    trie = _build_trie(seqs)
    leaf_idx = _leaf_idx_from_trie(trie, len(seqs))
    assert int(leaf_idx[0]) == int(leaf_idx[1])  # identical samples share leaf
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=10_000)
    pos0 = next(i for i, mb in enumerate(mbs) if 0 in mb)
    pos1 = next(i for i, mb in enumerate(mbs) if 1 in mb)
    assert pos0 == pos1, f"duplicate samples split: mb0={pos0} mb1={pos1}"


def test_mbs_groups_from_leaf_idx_leaf_ref_validation():
    """An orphan sample (leaf_idx=-1) raises; a ref to an internal node
    (strict-prefix sample) is accepted and groups without error."""
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    leaf_idx[1] = -1  # orphan: sample 1 has no leaf
    with pytest.raises(ValueError, match="no leaf assigned"):
        mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)
    internal_node = next(node for node in trie.nodes if node.children)
    leaf_idx[1] = internal_node.node_idx
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))


def test_mbs_groups_from_leaf_idx_budget_overrides_uid():
    """Uid atomicity is owned by DP balancing, not mbs grouping: rollouts of one
    prompt may split across micro-batches so the flat-token budget is always
    respected (budget 15 fits one leaf's path but not two rollouts of a prompt)."""
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=15)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    for mb in mbs:
        assert trie_group_flat_tokens(mb, trie) <= 15, f"micro-batch over budget: {trie_group_flat_tokens(mb, trie)}"


def _prep_td(samples, budget, trie, leaf_idx):
    """Tensordict for prepare_prefix_tree_micro_batches (dynamic bsz, prefix tree)."""
    n = len(samples)
    seq_len = max(len(s) for s in samples)
    input_ids = torch.zeros((n, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    for i, s in enumerate(samples):
        input_ids[i, : len(s)] = s
        attention_mask[i, : len(s)] = 1
    return tu.get_tensordict(
        tensor_dict={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "leaf_idx": leaf_idx,
        },
        non_tensor_dict={
            "prefix_tree": trie,
            "use_dynamic_bsz": True,
            "use_prefix_tree": True,
            "sp_size": 1,
            "force_group_size": 1,
            "max_token_len_per_gpu": budget,
        },
    )


def test_prepare_prefix_tree_micro_batches_attaches_subtrie():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    td = _prep_td(samples, 500, trie, leaf_idx)
    micro_batches, batch_idx_list = prepare_prefix_tree_micro_batches(td, sp_size=1)
    assert len(micro_batches) == len(batch_idx_list)
    for mb, mb_idx in zip(micro_batches, batch_idx_list, strict=False):
        subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
        assert subtree is not None, "prefix_tree_subtree not attached"
        mb_leaves = sorted(int(x) for x in mb["leaf_idx"].tolist())
        sub_leaves = sorted(subtree.leaf_node_ids)
        assert sub_leaves == mb_leaves, f"{sub_leaves} != {mb_leaves}"


def _leaf_entries(trie, leaf_idx):
    """leaf_entries list for calling _mbs_groups_dfs directly (pre-rebalance walk)."""
    leaf_to_positions: dict[int, list[int]] = {}
    for new_pos, leaf_fid in enumerate(leaf_idx.tolist()):
        leaf_to_positions.setdefault(int(leaf_fid), []).append(new_pos)
    entries = []
    for node in trie.nodes:
        positions = leaf_to_positions.get(node.node_idx)
        if positions is not None:
            entries.append((node, positions))
    return entries


def _mb_cost_spread(groups, trie):
    """max/min dedup flat-token cost over buckets (min floored at 1)."""
    costs = [trie_group_flat_tokens(g, trie) for g in groups]
    return max(costs) / max(min(c for c in costs if c > 0), 1)


def _fuzz_configs():
    # (n_prompts, rollout_n, prefix_len, resp_len, budget, seed)
    return [
        (8, 4, 64, 24, 1200, 1),
        (16, 4, 48, 20, 900, 2),
        (24, 4, 32, 16, 700, 3),
        (12, 8, 40, 12, 800, 4),
        (32, 4, 56, 28, 1500, 5),
        (20, 2, 100, 40, 1200, 6),
        (10, 4, 80, 60, 2000, 7),
        (40, 4, 24, 12, 600, 8),
        (16, 4, 64, 32, 450, 9),  # tight budget -> many buckets
        (14, 6, 50, 18, 1000, 10),
    ]


def test_balance_pass_fuzz_invariants():
    """Balance pass (_rebalance_groups via mbs_groups_from_leaf_idx) on fuzzed
    trees: every sample covered exactly once, every bucket's dedup flat-token
    cost within budget, and median max/min bucket cost materially below the raw
    DFS walk (the greedy cap-fill leaves a runt last bucket: cap, cap, ...,
    leftover; best-effort target is max/min <= 1.5)."""
    before, after = [], []
    for n_prompts, rollout_n, prefix_len, resp_len, budget, seed in _fuzz_configs():
        samples = _make_samples(n_prompts, rollout_n, prefix_len, resp_len, seed=seed)
        trie = _build_trie(samples)
        leaf_idx = _leaf_idx_from_trie(trie, len(samples))
        bal = mbs_groups_from_leaf_idx(leaf_idx, trie, budget)
        all_pos = sorted(p for g in bal for p in g)
        assert all_pos == list(range(len(samples))), (
            f"config {seed}: coverage broken — missing={set(range(len(samples))) - set(all_pos)}"
        )
        for gi, g in enumerate(bal):
            cost = trie_group_flat_tokens(g, trie)
            assert cost <= budget, f"config {seed}: bucket {gi} cost {cost} > budget {budget}"
        raw = _mbs_groups_dfs(_leaf_entries(trie, leaf_idx), budget)
        if len(raw) >= 2:
            before.append(_mb_cost_spread(raw, trie))
            after.append(_mb_cost_spread(bal, trie))
    assert after, "no multi-bucket configs generated"
    assert statistics.median(after) <= 1.5, f"median balance {statistics.median(after):.2f} > 1.5 after rebalance"
    assert statistics.median(after) < statistics.median(before), "rebalance did not improve median balance"


def test_pp2_divisibility_halves_without_singletons():
    """Roundup to a PP2 multiple must halve the largest bucket, never peel a
    1-sample runt (the old `g[:-1]/[g[-1]]` peel created singleton mbs that
    idle their pipeline slot), and the resulting mb count is exactly
    roundup_divisible(budget-derived count, 2) — no overshoot."""
    for seed, budget in [(1, 1200), (2, 900), (3, 700), (6, 1200), (7, 2000), (9, 450)]:
        samples = _make_samples(8, 4, prefix_len=64, resp_len=24, seed=seed)
        trie = _build_trie(samples)
        leaf_idx = _leaf_idx_from_trie(trie, len(samples))
        td = _prep_td(samples, budget, trie, leaf_idx)
        _, mbs = prepare_prefix_tree_micro_batches(td, sp_size=1, num_batches_divided_by=2)
        assert len(mbs) % 2 == 0, f"seed {seed}: mb count not PP2-divisible"
        for g in mbs:
            assert len(g) > 1, f"seed {seed}: singleton micro-bucket {g} (peel-one bug)"
        assert sorted(p for g in mbs for p in g) == list(range(len(samples))), f"seed {seed}: coverage broken"
        base = mbs_groups_from_leaf_idx(leaf_idx, trie, budget)
        expected = roundup_divisible(len(base), 2)
        assert len(mbs) == expected, (
            f"seed {seed}: got {len(mbs)} mbs, expected {expected} (base {len(base)} rounded up to PP2)"
        )


# ---- 2-rank DP micro-batch count equalization ----
# The tree path must produce the SAME number of micro-batches on every DP rank
# (same_micro_num_in_dp), otherwise the DP gradient collective desyncs. When one
# rank has fewer samples than the DP-max micro-batch count, it cannot split down
# to the target count — it must raise (loudly, on every rank, so no peer hangs),
# not warn-and-continue with unequal counts.


class _CPUDevice:
    def current_device(self):
        return "cpu"


def _fixed_mbs_td(samples, trie, leaf_idx, mbs):
    """Tensordict for prepare_prefix_tree_micro_batches (fixed mbs, prefix tree)."""
    n = len(samples)
    seq_len = max(len(s) for s in samples)
    input_ids = torch.zeros((n, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    for i, s in enumerate(samples):
        input_ids[i, : len(s)] = s
        attention_mask[i, : len(s)] = 1
    return tu.get_tensordict(
        tensor_dict={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "leaf_idx": leaf_idx,
        },
        non_tensor_dict={
            "prefix_tree": trie,
            "use_dynamic_bsz": False,
            "use_prefix_tree": True,
            "micro_batch_size_per_gpu": mbs,
            "force_group_size": 1,
        },
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _equalize_worker(rank, world_size, port, n_local_per_rank, result_queue):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    pt_dynamic.get_torch_device = _CPUDevice
    try:
        all_samples = _make_samples(4, 4, prefix_len=20, resp_len=10, seed=42)
        trie = _build_trie(all_samples)
        # deterministic per-rank slice: rank r takes a contiguous block of samples
        start = sum(n_local_per_rank[:rank])
        local_ids = list(range(start, start + n_local_per_rank[rank]))
        leaf_idx = _leaf_idx_from_trie(trie, len(all_samples))[local_ids]
        data = _fixed_mbs_td([all_samples[i] for i in local_ids], trie, leaf_idx, mbs=4)
        _, groups = prepare_prefix_tree_micro_batches(
            data, sp_size=1, dp_group=dist.group.WORLD, same_micro_num_in_dp=True
        )
        result_queue.put(("ok", rank, len(groups)))
    except ValueError as e:
        result_queue.put(("raise", rank, str(e)))
    except Exception as e:  # pragma: no cover - surfaces unexpected failure
        result_queue.put(("error", rank, repr(e)))
    finally:
        dist.destroy_process_group()


def _run_two_rank(n_local_per_rank):
    port = _free_port()
    # spawn-context queue: mp.Queue() defaults to the ambient context, and a
    # fork-context SemLock cannot be shared with spawn workers (Ray sets fork).
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_equalize_worker, args=(2, port, n_local_per_rank, q), nprocs=2, join=True)
    return [q.get(timeout=30) for _ in range(2)]


def test_equalize_reaches_same_count_when_feasible():
    """Both ranks have 4 samples, mbs=4 -> each makes 1 group. n_mb=1. No raise, equal counts."""
    results = _run_two_rank([4, 4])
    assert sorted(r[0] for r in results) == ["ok", "ok"], f"unexpected results: {results}"
    counts = [r[2] for r in results]
    assert counts[0] == counts[1], f"unequal counts across ranks: {counts}"


def test_starved_rank_raises_on_both_ranks():
    """rank0: 8 samples / mbs 4 = 2 groups; rank1: 1 sample / mbs 4 = 1 group.
    n_mb = 2, min_samples = 1 -> 2 > 1 -> infeasible -> both ranks raise."""
    results = _run_two_rank([8, 1])
    kinds = sorted(r[0] for r in results)
    assert kinds == ["raise", "raise"], f"expected both ranks to raise, got: {results}"
    for r in results:
        assert "cannot equalize micro-batch count" in r[2], f"unexpected message: {r[2]}"
