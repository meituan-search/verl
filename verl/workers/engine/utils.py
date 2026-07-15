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

import os
import random

import numpy as np
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import is_npu_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches, restore_dynamic_batch


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if is_npu_available:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if is_npu_available:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Prepare micro batches from data.
    """
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    use_prefix_tree = tu.get_non_tensor_data(data=data, key="use_prefix_tree", default=False)
    sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)

    force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)

    if use_dynamic_bsz and use_prefix_tree:
        from verl.utils.prefix_tree.dynamic import (
            greedy_build_tries,
            mbs_groups_from_trie,
            prepare_prefix_tree_micro_batches,
            trie_group_flat_tokens,
        )

        # Build trie once, thread through batch metadata.
        input_ids = data["input_ids"]
        seqs = [t.tolist() for t in input_ids.unbind()]
        total_raw = sum(len(s) for s in seqs)
        max_tokens = total_raw * 10
        tries, num_tokens = greedy_build_tries(seqs, max_tokens_per_tree=max_tokens)
        if tries and total_raw > 0:
            trie = tries[0]  # TODO: use PrefixTrie
            flat = num_tokens[0]
            tu.assign_non_tensor(data, prefix_tree=trie)  # TODO: use PrefixTrie

            # Build leaf_ids: sample_idx → flat position of that sample's leaf node.
            # TODO: move into PrefixTrie constructor once TrieNode gets flat_idx.
            import numpy as np
            _leaf_ids = np.full(len(seqs), -1, dtype=np.int64)
            for _flat_idx, _node in enumerate(trie.nodes):
                if not _node.children:  # leaf node
                    for _seq_id in _node.sequence_ids:
                        _leaf_ids[_seq_id] = _flat_idx
            tu.assign_non_tensor(data, prefix_tree_leaf_ids=_leaf_ids)

            # Compute prefix-tree metrics from the already-built trie — correct
            # sequence lengths (NestedTensor, no padding) and no trie rebuild.
            max_token_len_per_gpu = tu.get_non_tensor_data(data, "max_token_len_per_gpu", default=None)
            pt_metrics = {
                "prefix_tree/global_shared_ratio": 1.0 - flat / total_raw,
                "prefix_tree/flat_tokens": flat,
                "prefix_tree/raw_tokens": total_raw,
            }
            if max_token_len_per_gpu is not None:
                groups = mbs_groups_from_trie(trie, max_token_len_per_gpu * sp_size)  # TODO: use PrefixTrie
                pt_metrics["prefix_tree/avg_mbs"] = len(seqs) / len(groups) if groups else 0.0
                ratios = [
                    1.0 - trie_group_flat_tokens(g, trie) / sum(len(seqs[i]) for i in g)  # TODO: use PrefixTrie
                    for g in groups
                    if sum(len(seqs[i]) for i in g) > 0
                ]
                if ratios:
                    pt_metrics["prefix_tree/micro_batch_shared_ratio"] = sum(ratios) / len(ratios)
            tu.assign_non_tensor(data, prefix_tree_metrics=pt_metrics)

        micro_batches, batch_idx_list = prepare_prefix_tree_micro_batches(
            data,
            sp_size=sp_size,
            dp_group=dp_group,
            same_micro_num_in_dp=same_micro_num_in_dp,
            num_batches_divided_by=num_batches_divided_by,
        )

    elif use_dynamic_bsz:
        assert "max_token_len_per_gpu" in data.keys(), "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
        max_token_len_per_gpu = data["max_token_len_per_gpu"]
        max_token_len = max_token_len_per_gpu * sp_size
        micro_batches, batch_idx_list = rearrange_micro_batches(
            data,
            max_token_len=max_token_len,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
            force_group_size=force_group_size,
        )
    else:
        # When use_prefix_tree=True, the batch is already DFS-ordered by the
        # global _balance_batch step (reorder_and_balance_for_prefix_tree).
        # Contiguous micro-batch slices then naturally group same-prompt sequences
        # together so the prefix tree can find shared prefixes.  No local re-sort needed.
        total_data_size = len(data)
        micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
        assert total_data_size % (force_group_size * micro_batch_size_per_gpu) == 0, (
            "data size must be divisible by force_group_size * micro_batch_size_per_gpu"
        )
        mbs = micro_batch_size_per_gpu * force_group_size
        n_micro = total_data_size // mbs
        micro_batches = tu.chunk_tensordict(data, n_micro)
        batch_idx_list = None
    return micro_batches, batch_idx_list


def postprocess_batch_func(output_lst, indices, data: TensorDict):
    """postprocess the output of a forward_backward_batch.
    output_lst is a list of dict containing outputs for each micro-batch
    reorder entropy and outputs. Return None for other pp ranks
    only on last rank. It should be on every tp rank

    each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    """

    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    assert pad_mode == DatasetPadMode.NO_PADDING, "postprocess_batch_func only support NO_PADDING pad_mode"

    # losses_reduced is a list of dict containing outputs for each micro-batch
    # reorder entropy and outputs. Return None for other pp ranks
    # only on last rank. It should be on every tp rank

    # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    # We perform reverse

    model_output = {}
    losses = []
    aggregated_metrics = {}

    # model output
    for o in output_lst:
        if "model_output" in o:
            for key, val in o["model_output"].items():
                if key not in model_output:
                    model_output[key] = []
                model_output[key].append(val)

    # concat results from micro batches
    for key, val in model_output.items():
        if pad_mode == DatasetPadMode.NO_PADDING:
            tensors = [tensor for nt in model_output[key] for tensor in nt.unbind()]
            model_output[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        # reverse with dynamic bsz
        if use_dynamic_bsz:
            model_output[key] = restore_dynamic_batch(model_output[key], indices)

    # loss
    for o in output_lst:
        if "loss" in o:
            losses.append(o["loss"])

    # metrics
    for o in output_lst:
        if "metrics" in o:
            metrics = o["metrics"]
            append_to_dict(aggregated_metrics, metrics)

    output = {
        "model_output": model_output,
        "loss": losses,
        "metrics": aggregated_metrics,
    }

    return output
