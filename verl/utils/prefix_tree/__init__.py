# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from . import balancing, trainer  # noqa: F401 — sub-module registrations
from .dynamic import (
    TrieNode,
    build_mini_batch_prefix_groups,
    build_tree_dynamic,
    compute_prefix_sharing_ratio,
    compute_prefix_tree_metrics,
    convert_trie_to_tree_node,
    dfs_leaf_order,
    dfs_micro_batch_groups,
    greedy_build_tries,
    prepare_prefix_tree_micro_batches,
)
from .hash_based import _hash_prefix as hash_prefix
from .magi import (
    PrefixTreeMagiBatch,
    build_prefix_segments_single_turn,
    build_prefix_tree_batch,
    build_prefix_tree_micro_batch,
    forward_prefix_tree,
    get_prefix_tree_kwargs,
    restore_flat_to_nested,
    strip_prefix_tree_args,
)
from .utils import TreeNode, build_layout_from_tree_node, build_prefix_tree_params, longest_common_prefix_length

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
    "greedy_build_tries",
    "prepare_prefix_tree_micro_batches",
    # hash_based
    "hash_prefix",
    # magi
    "PrefixTreeMagiBatch",
    "build_prefix_segments_single_turn",
    "build_prefix_tree_batch",
    "build_prefix_tree_micro_batch",
    "forward_prefix_tree",
    "get_prefix_tree_kwargs",
    "restore_flat_to_nested",
    "strip_prefix_tree_args",
    # utils
    "TreeNode",
    "build_layout_from_tree_node",
    "build_prefix_tree_params",
    "longest_common_prefix_length",
]
