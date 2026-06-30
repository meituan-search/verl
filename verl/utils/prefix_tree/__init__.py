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

from . import trainer  # noqa: F401 — sub-module registrations
from .dynamic import (
    TrieNode,
    build_mini_batch_prefix_groups,
    build_tree_dynamic,
    compute_prefix_sharing_ratio,
    compute_prefix_tree_metrics,
    convert_trie_to_tree_node,
    dfs_leaf_order,
    dfs_micro_batch_groups,
    get_dfs_balanced_partitions,
    get_prefix_balanced_partitions,
    greedy_build_tries,
    mbs_groups_from_trie,
    mbs_groups_from_uid,
    prepare_prefix_tree_micro_batches,
    prune_trie,
    reorder_and_balance_for_prefix_tree,
    trie_dfs_leaf_order,
    trie_to_leaf_ids,
)
from .magi import (
    PrefixTreeMagiBatch,
    build_prefix_tree_batch,
    build_prefix_tree_micro_batch,
    fuse_try_forward_prefix_tree,
    fuse_undispatch_and_expand_hidden,
    get_prefix_tree_kwargs,
    restore_flat_to_nested,
    strip_prefix_tree_args,
    unfuse_forward_prefix_tree,
    unfuse_try_forward_prefix_tree,
)
from .utils import (
    PrefixTreeParams,
    RangeSpec,
    build_layout_from_tree_node,
    longest_common_prefix_length,
)

__all__ = [
    # dynamic
    "TrieNode",
    "build_mini_batch_prefix_groups",
    "build_tree_dynamic",
    "compute_prefix_sharing_ratio",
    "compute_prefix_tree_metrics",
    "convert_trie_to_tree_node",
    "dfs_leaf_order",
    "dfs_micro_batch_groups",
    "get_dfs_balanced_partitions",
    "get_prefix_balanced_partitions",
    "greedy_build_tries",
    "mbs_groups_from_trie",
    "mbs_groups_from_uid",
    "prepare_prefix_tree_micro_batches",
    "prune_trie",
    "reorder_and_balance_for_prefix_tree",
    "trie_dfs_leaf_order",
    "trie_to_leaf_ids",
    # magi
    "PrefixTreeMagiBatch",
    "build_prefix_tree_batch",
    "build_prefix_tree_micro_batch",
    "fuse_try_forward_prefix_tree",
    "fuse_undispatch_and_expand_hidden",
    "get_prefix_tree_kwargs",
    "restore_flat_to_nested",
    "strip_prefix_tree_args",
    "unfuse_forward_prefix_tree",
    "unfuse_try_forward_prefix_tree",
    # utils
    "PrefixTreeParams",
    "RangeSpec",
    "build_layout_from_tree_node",
    "longest_common_prefix_length",
]
