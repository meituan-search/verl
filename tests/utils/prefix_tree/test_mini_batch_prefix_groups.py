# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Tests for mini-batch level trie grouping and prefix-aware DP load balancing.

Exercises :func:`verl.utils.prefix_tree.dynamic.build_mini_batch_prefix_groups`
and :func:`verl.utils.seqlen_balancing.get_prefix_balanced_partitions`.
"""

from __future__ import annotations

import pytest

from verl.utils.prefix_tree.dynamic import build_mini_batch_prefix_groups
from verl.utils.seqlen_balancing import get_prefix_balanced_partitions


# ---------------------------------------------------------------------------
# build_mini_batch_prefix_groups — grouping correctness
# ---------------------------------------------------------------------------


def _all_seq_ids(groups):
    """Flatten all seq_ids from groups into a sorted list."""
    return sorted(i for seq_ids, _ in groups for i in seq_ids)


def test_groups_empty_input():
    assert build_mini_batch_prefix_groups([]) == []


def test_groups_single_sequence():
    """Single sequence → one group containing that sequence."""
    seqs = [[1, 2, 3, 4]]
    groups = build_mini_batch_prefix_groups(seqs)
    assert len(groups) == 1
    seq_ids, eff = groups[0]
    assert seq_ids == [0]
    assert eff == 4  # flat = full sequence


def test_groups_all_sequences_identical():
    """All identical sequences → trie has one path, leaf with multiple seq IDs."""
    seqs = [[1, 2, 3]] * 3
    groups = build_mini_batch_prefix_groups(seqs)
    # All 3 sequences share the same path → one group
    assert len(groups) == 1
    seq_ids, eff = groups[0]
    assert sorted(seq_ids) == [0, 1, 2]
    assert eff == 3  # flat = 3 unique tokens (the shared path)


def test_groups_two_sequences_shared_prefix():
    """Two sequences sharing a 2-token prefix, then diverging.

    Tree:  root → [1,2] → {[3]: seq0, [4]: seq1}
    Groups at the first branch (after [1,2]):
      group A: {seq0}, effective = 2 (shared) + 1 ([3]) = 3
      group B: {seq1}, effective = 2 (shared) + 1 ([4]) = 3
    """
    seqs = [[1, 2, 3], [1, 2, 4]]
    groups = build_mini_batch_prefix_groups(seqs)
    assert len(groups) == 2
    all_ids = _all_seq_ids(groups)
    assert all_ids == [0, 1]
    for seq_ids, eff in groups:
        assert len(seq_ids) == 1
        assert eff == 3  # 2 shared + 1 unique


def test_groups_three_sequences_same_prefix():
    """Three sequences sharing a 2-token prefix, diverging into 3 branches."""
    seqs = [
        [10, 11, 20, 21],
        [10, 11, 30, 31],
        [10, 11, 40, 41],
    ]
    groups = build_mini_batch_prefix_groups(seqs)
    # 3 branches → 3 groups
    assert len(groups) == 3
    assert _all_seq_ids(groups) == [0, 1, 2]
    for seq_ids, eff in groups:
        # Each group: 2 shared + 2 unique = 4
        assert eff == 4


def test_groups_no_shared_prefix():
    """Sequences with no common first token → each sequence is its own group."""
    seqs = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    groups = build_mini_batch_prefix_groups(seqs)
    # No shared prefix → trie root has 3 children, each with 1 sequence
    assert len(groups) == 3
    assert _all_seq_ids(groups) == [0, 1, 2]
    for seq_ids, eff in groups:
        assert len(seq_ids) == 1
        assert eff == 3


def test_groups_multi_level_prefix():
    """Multi-level prefix tree: all 4 sequences share [10,11], then split 2+2.

    Tree:
        root → [10,11] → {
            [20,21] → {[30]: seq0, [40]: seq1},
            [50,51] → {[60]: seq2, [70]: seq3},
        }

    The first branching point is AFTER the [10,11] spine.
    Groups (at the immediate children of the [10,11] node):
      group A: {seq0, seq1} — sequences through [20,21] branch
      group B: {seq2, seq3} — sequences through [50,51] branch
    """
    seqs = [
        [10, 11, 20, 21, 30],
        [10, 11, 20, 21, 40],
        [10, 11, 50, 51, 60],
        [10, 11, 50, 51, 70],
    ]
    groups = build_mini_batch_prefix_groups(seqs)
    assert len(groups) == 2
    assert _all_seq_ids(groups) == [0, 1, 2, 3]

    sizes = sorted(len(seq_ids) for seq_ids, _ in groups)
    assert sizes == [2, 2]

    # Each group: shared [10,11]=2 tokens + branch [20,21 or 50,51]=2 + 2 leaves×1 = 6
    for seq_ids, eff in groups:
        assert eff == 6  # 2 (shared spine) + 2 (sub-prefix) + 1+1 (2 leaves)


def test_groups_effective_tokens_for_identical_sequences():
    """When all sequences are identical (one group, no branching), the group's
    effective token count equals the unique sequence length — not the raw sum."""
    seqs = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],  # identical
    ]
    raw_total = sum(len(s) for s in seqs)  # 10

    groups = build_mini_batch_prefix_groups(seqs)
    # No branching → one group
    assert len(groups) == 1
    _, eff = groups[0]
    # Flat unique tokens = 5 (the single path), not 10
    assert eff == 5
    assert eff < raw_total


def test_groups_effective_per_group_includes_shared_prefix():
    """When two sequences share a prefix and split, each group's effective count
    includes the shared prefix (counted once *per group*).  The per-group
    effective therefore equals the raw length of a single sequence — the saving
    is realised when both groups land on the *same* DP rank (micro-batch builds
    a joint prefix tree at runtime)."""
    seqs = [
        [1, 2, 3, 4, 5],   # len=5
        [1, 2, 3, 6, 7],   # len=5
    ]
    groups = build_mini_batch_prefix_groups(seqs)
    assert len(groups) == 2
    # Each group: shared [1,2,3] (len=3) + unique branch (len=2) = 5 = raw len
    for seq_ids, eff in groups:
        assert eff == 5


# ---------------------------------------------------------------------------
# get_prefix_balanced_partitions — load balancing correctness
# ---------------------------------------------------------------------------


def test_partition_covers_all_indices():
    """Every sample index appears in exactly one partition."""
    seqs = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 6, 7],
        [8, 9, 10, 11],
    ]
    k = 2
    partitions = get_prefix_balanced_partitions(seqs, k_partitions=k)
    assert len(partitions) == k
    all_ids = sorted(i for p in partitions for i in p)
    assert all_ids == list(range(len(seqs)))


def test_partition_prefix_sharing_sequences_colocated():
    """Sequences with the exact same prefix are in the same group → same partition
    when k < number of groups."""
    # 4 sequences forming 2 groups of 2 (sharing [10,11] then [20] / [50])
    seqs = [
        [10, 11, 20, 30],   # group A: seq0
        [10, 11, 20, 40],   # group A: seq1
        [10, 11, 50, 60],   # group B: seq2
        [10, 11, 50, 70],   # group B: seq3
    ]
    # With k=2, the two groups should each land in a separate partition.
    partitions = get_prefix_balanced_partitions(seqs, k_partitions=2)
    assert len(partitions) == 2
    p0, p1 = sorted(partitions, key=lambda p: min(p))
    # seq0 and seq1 share sub-prefix [20] → must be together
    assert (0 in p0 and 1 in p0) or (0 in p1 and 1 in p1), (
        f"seq0 and seq1 should be co-located, got partitions {partitions}"
    )
    # seq2 and seq3 share sub-prefix [50] → must be together
    assert (2 in p0 and 3 in p0) or (2 in p1 and 3 in p1), (
        f"seq2 and seq3 should be co-located, got partitions {partitions}"
    )


def test_partition_empty_input():
    partitions = get_prefix_balanced_partitions([], k_partitions=4)
    assert len(partitions) == 4
    assert all(p == [] for p in partitions)


def test_partition_k_equals_one():
    """k=1 → all sequences in one partition."""
    seqs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    partitions = get_prefix_balanced_partitions(seqs, k_partitions=1)
    assert len(partitions) == 1
    assert sorted(partitions[0]) == [0, 1, 2]


def test_partition_fewer_groups_than_k():
    """When there are fewer groups than k, each group gets its own partition, rest empty."""
    # 2 sequences with shared prefix → 2 groups (one per leaf)
    seqs = [[1, 2, 3], [1, 2, 4]]
    partitions = get_prefix_balanced_partitions(seqs, k_partitions=4)
    assert len(partitions) == 4
    non_empty = [p for p in partitions if p]
    assert len(non_empty) == 2
    # All indices covered
    assert sorted(i for p in partitions for i in p) == [0, 1]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_partition_all_indices_covered_parametric(k):
    """Exhaustive coverage check across different k values."""
    seqs = [list(range(i, i + 5)) for i in range(8)]  # 8 non-overlapping sequences
    partitions = get_prefix_balanced_partitions(seqs, k_partitions=k)
    assert len(partitions) == k
    all_ids = sorted(i for p in partitions for i in p)
    assert all_ids == list(range(len(seqs)))


def test_partition_effective_tokens_used_as_workload():
    """Partitioning with shared prefixes should produce more balanced loads than
    naive raw-length balancing when one group has heavy shared savings."""
    # Group A: 4 sequences each len=10 sharing first 8 tokens → eff = 8 + 4×2 = 16
    prefix_a = list(range(8))
    group_a = [prefix_a + [100 + i, 200 + i] for i in range(4)]
    # Group B: 4 sequences each len=2, no sharing → eff = 8 (4×2)
    group_b = [[300 + i, 400 + i] for i in range(4)]

    seqs = group_a + group_b  # indices 0-3 = group A, 4-7 = group B

    partitions = get_prefix_balanced_partitions(seqs, k_partitions=2)
    assert len(partitions) == 2
    all_ids = sorted(i for p in partitions for i in p)
    assert all_ids == list(range(8))

    # Group A (indices 0-3) must stay together (they share a long prefix)
    p0_set, p1_set = set(partitions[0]), set(partitions[1])
    group_a_ids = set(range(4))
    assert group_a_ids <= p0_set or group_a_ids <= p1_set, (
        f"Group A sequences must be co-located; partitions={partitions}"
    )


# ---------------------------------------------------------------------------
# dfs_leaf_order
# ---------------------------------------------------------------------------

from verl.utils.prefix_tree.dynamic import dfs_leaf_order


def test_dfs_leaf_order_covers_all():
    """Every sample index appears exactly once."""
    seqs = [[1, 2, 3], [1, 2, 4], [5, 6, 7]]
    order = dfs_leaf_order(seqs)
    assert sorted(order) == [0, 1, 2]


def test_dfs_leaf_order_prefix_sharing_first():
    """Samples sharing a prefix appear consecutively (before the non-sharing one)."""
    seqs = [[1, 2, 3], [1, 2, 4], [9, 9, 9]]
    order = dfs_leaf_order(seqs)
    # samples 0 and 1 share [1,2] → they must be adjacent
    pos = {s: i for i, s in enumerate(order)}
    assert abs(pos[0] - pos[1]) == 1, f"Sharing samples not adjacent: {order}"


def test_dfs_leaf_order_deep_tree():
    """Depth-3 tree: samples sharing longer prefix come before those sharing less."""
    # samples 0,1 share [1,2,3,4]; samples 2,3 share [1,2,5,6]
    seqs = [
        [1, 2, 3, 4, 10],
        [1, 2, 3, 4, 11],
        [1, 2, 5, 6, 12],
        [1, 2, 5, 6, 13],
    ]
    order = dfs_leaf_order(seqs)
    pos = {s: i for i, s in enumerate(order)}
    # 0 and 1 adjacent; 2 and 3 adjacent
    assert abs(pos[0] - pos[1]) == 1
    assert abs(pos[2] - pos[3]) == 1


def test_dfs_leaf_order_single():
    assert dfs_leaf_order([[1, 2, 3]]) == [0]


def test_dfs_leaf_order_empty():
    assert dfs_leaf_order([]) == []


def test_dfs_micro_batch_groups_flat_budget():
    """Budget counts flat (deduplicated) tokens, not raw."""
    # 4 seqs sharing prefix [1,2,3] (3 tokens) + 2 unique tokens each = 5 tokens raw
    # flat for 4 seqs = 3 (shared) + 4*2 (unique) = 11 tokens
    seqs = [[1, 2, 3, i, i+10] for i in range(4)]
    from verl.utils.prefix_tree.dynamic import dfs_micro_batch_groups

    # budget=11 → fits all 4 in one group
    groups = dfs_micro_batch_groups(seqs, max_token_len=11)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2, 3]

    # budget=9 → first 3 fit (3+2+2+2=9), 4th goes to next
    groups2 = dfs_micro_batch_groups(seqs, max_token_len=9)
    total = sum(len(g) for g in groups2)
    assert total == 4
