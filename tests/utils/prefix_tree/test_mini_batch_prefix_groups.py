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

"""Tests for trie-based DFS leaf ordering, micro-batch grouping, and flat-token workload estimation."""

from __future__ import annotations

from verl.utils.prefix_tree.dynamic import (
    dfs_leaf_order,
    greedy_build_tries,
    mbs_groups_from_trie,
    trie_group_flat_tokens,
)

# ---------------------------------------------------------------------------
# dfs_leaf_order
# ---------------------------------------------------------------------------


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
    seqs = [[1, 2, 3, i, i + 10] for i in range(4)]
    from verl.utils.prefix_tree.dynamic import dfs_micro_batch_groups

    # budget=11 → fits all 4 in one group
    groups = dfs_micro_batch_groups(seqs, max_token_len=11)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2, 3]

    # budget=9 → first 3 fit (3+2+2+2=9), 4th goes to next
    groups2 = dfs_micro_batch_groups(seqs, max_token_len=9)
    total = sum(len(g) for g in groups2)
    assert total == 4


# ---------------------------------------------------------------------------
# trie_group_flat_tokens — mbs workload estimator
# ---------------------------------------------------------------------------


def _build_trie(sequences):
    max_tokens = sum(len(s) for s in sequences) * 10
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens)
    return tries[0]


def test_trie_group_flat_tokens_all_sequences():
    """Full group equals the whole trie's flat token count.

    Trie structure for [[1,2,3,10],[1,2,3,11],[1,2,4,12]]:
      root → [1,2](2) → [3](1) → [10](1)
                                → [11](1)
                      → [4,12](2)
    flat = 2+1+1+1+2 = 7
    """
    seqs = [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12]]
    trie = _build_trie(seqs)
    flat = trie_group_flat_tokens(list(range(len(seqs))), trie)
    assert flat == 7


def test_trie_group_flat_tokens_subgroup():
    """Subgroup pays for its minimal sub-trie including shared ancestors."""
    # Same trie: root → [1,2](2) → [3](1) → [10](1)
    #                                       → [11](1)
    #                             → [4,12](2)
    seqs = [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12]]
    trie = _build_trie(seqs)
    # Group {0,1}: [1,2]=2 + [3]=1 + [10]=1 + [11]=1 = 5
    flat_01 = trie_group_flat_tokens([0, 1], trie)
    assert flat_01 == 5
    # Group {2}: [1,2]=2 + [4,12]=2 = 4  (still pays for the shared [1,2] root edge)
    flat_2 = trie_group_flat_tokens([2], trie)
    assert flat_2 == 4


def test_trie_group_flat_tokens_after_mbs_sort():
    """After mbs are sorted/reordered for DP assignment the trie is unchanged
    and trie_group_flat_tokens returns the same value for each group."""
    seqs = [[1, 2, 3, i] for i in range(8)]  # 8 seqs sharing [1,2,3]
    trie = _build_trie(seqs)
    mbs_groups = mbs_groups_from_trie(trie, max_token_len=10)

    # Record flat tokens for each mbs before any reordering.
    flat_before = [trie_group_flat_tokens(g, trie) for g in mbs_groups]

    # Sort mbs by flat tokens — the trie object must remain intact.
    sorted_groups = sorted(mbs_groups, key=lambda g: trie_group_flat_tokens(g, trie))
    flat_after = [trie_group_flat_tokens(g, trie) for g in sorted_groups]

    assert sorted(flat_before) == sorted(flat_after), "Flat token counts must be stable after mbs reordering"
