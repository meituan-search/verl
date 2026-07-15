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

"""Prefix-tree load balancing helpers consumed by trainers."""

from __future__ import annotations

import torch


def get_prefix_balanced_partitions(
    data,
    config_or_data: dict,
    dp_size: int,
    *,
    attention_mask=None,
    contiguous_partitions: bool = False,
):
    """Re-order batch in DFS trie order and return balanced partitions.

    When *config_or_data* has ``use_prefix_tree=True``, this function:
    1.  Strips padding from each row via *attention_mask* (or raw slices).
    2.  Calls :func:`dfs_leaf_order` to get a DFS pre-order permutation.
    3.  Reorders *data* in-place (``data.reorder`` or ``index_select_tensor_dict``
        depending on the container type).
    4.  Returns per-rank partition lists.

    If ``use_prefix_tree`` is ``False`` returns ``None`` so the caller falls
    back to its normal balancing logic.

    Args:
        data: ``DataProto`` or ``TensorDict`` containing ``input_ids``.
        config_or_data: config dict or object supporting ``.get(key, default)``.
        dp_size: number of DP ranks.
        attention_mask: optional 2-D tensor.  When given, padding tokens are
            stripped by the mask so left-padded sequences don't break the trie.
        contiguous_partitions: if ``True`` use simple contiguous slicing
            (preserves DFS order).  If ``False`` use KK-style balanced
            partitions with ``equal_size=True``.

    Returns:
        ``(global_partition_lst, global_seqlen_lst)`` or ``None``.
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return None

    from verl.utils.prefix_tree.dynamic import dfs_leaf_order

    batch_size = data.batch["input_ids"].shape[0] if hasattr(data, "batch") else len(data["input_ids"])
    _ids = data.batch["input_ids"] if hasattr(data, "batch") else data["input_ids"]
    _mask = attention_mask or (data.batch.get("attention_mask", None) if hasattr(data, "batch") else None)

    if _mask is not None:
        seqs = [_ids[i][_mask[i].bool()].tolist() or [0] for i in range(batch_size)]
    else:
        seqs = [_ids[i].tolist() for i in range(batch_size)]

    dfs_order = dfs_leaf_order(seqs)
    if len(dfs_order) < batch_size:
        missing = [i for i in range(batch_size) if i not in set(dfs_order)]
        dfs_order = dfs_order + missing

    if hasattr(data, "reorder"):
        data.reorder(torch.tensor(dfs_order))
    else:
        from verl.utils import tensordict_utils as tu

        data = tu.index_select_tensor_dict(data, torch.tensor(dfs_order))

    # Recompute sequence lengths after DFS reorder
    if hasattr(data, "batch") and "attention_mask" in data.batch:
        global_seqlen_lst = data.batch["attention_mask"].view(batch_size, -1).sum(-1)
    else:
        global_seqlen_lst = torch.Tensor([item.size()[0] for item in data["input_ids"]])

    if contiguous_partitions:
        per_rank = batch_size // dp_size
        partition_lst = [list(range(i * per_rank, (i + 1) * per_rank)) for i in range(dp_size)]
    else:
        from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions

        partition_lst = get_seqlen_balanced_partitions(
            calculate_workload(global_seqlen_lst), k_partitions=dp_size, equal_size=True
        )

    return partition_lst, global_seqlen_lst, data


def _is_prefix_tree_enabled(config_or_data) -> bool:
    if isinstance(config_or_data, dict):
        return config_or_data.get("use_prefix_tree", False)
    return getattr(config_or_data, "use_prefix_tree", False)
