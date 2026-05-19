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


def _reorder_by_prefix_sharing(data: TensorDict, mbs: int) -> TensorDict:
    """Reorder samples within the mini-batch to maximize prefix sharing per micro-batch.

    Groups samples by their root prefix hash (first entry of prefix_segments) so that
    samples sharing the same prefix end up in the same micro-batch. Within each group,
    order is preserved (no within-group shuffle).

    Args:
        data: TensorDict of shape [N] containing the full mini-batch.
        mbs: micro_batch_size_per_gpu — determines group target size.

    Returns:
        Reordered TensorDict of same shape [N].
    """
    prefix_segments = tu.get_non_tensor_data(data, key="prefix_segments", default=None)
    if prefix_segments is None:
        return data  # no prefix info — return unchanged

    N = len(data)
    # Extract the full hash sequence for each sample (all turns, not just turn-1).
    # Sorting lexicographically on this sequence ensures that samples sharing the
    # longest common prefix end up adjacent — consecutive mbs windows get max sharing.
    seg_keys = []
    for i in range(N):
        segs = prefix_segments[i]
        if hasattr(segs, 'data'):
            segs = segs.data
        if segs and len(segs) > 0:
            seg_keys.append(tuple(h for h, _ in segs))
        else:
            seg_keys.append((i,))  # no hash → unique key, preserves original order

    # Sort indices by full hash-tuple so siblings (sharing deepest prefix) are adjacent
    new_order = sorted(range(N), key=lambda i: seg_keys[i])

    if len(new_order) != N:
        return data  # safety fallback

    from verl.utils.tensordict_utils import index_select_tensor_dict
    return index_select_tensor_dict(data, new_order)


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
    sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)
    force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)

    # When prefix-tree is active, reorder mini-batch samples so that
    # prefix-sharing samples are co-located within the same micro-batch.
    use_prefix_tree = tu.get_non_tensor_data(data=data, key="use_prefix_tree", default=False)
    if use_prefix_tree and not use_dynamic_bsz:
        mbs = tu.get_non_tensor_data(data=data, key="micro_batch_size_per_gpu", default=1)
        data = _reorder_by_prefix_sharing(data, mbs)

    if use_dynamic_bsz:
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
        total_data_size = len(data)
        micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
        assert total_data_size % (force_group_size * micro_batch_size_per_gpu) == 0, (
            "data size must be divisible by force_group_size * micro_batch_size_per_gpu"
        )
        micro_batches = tu.chunk_tensordict(data, total_data_size // (micro_batch_size_per_gpu * force_group_size))
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
