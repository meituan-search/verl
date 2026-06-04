# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Optional

import torch
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.megatron_utils import unwrap_model
from verl.workers.config import MtpConfig

from .util import (
    build_vlm_attn_mask_bshd,
    build_vlm_attn_mask_thd,
    postprocess_bshd,
    postprocess_bshd_engine,
    postprocess_packed_seqs,
    postprocess_thd_engine,
    preprocess_bshd,
    preprocess_bshd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)


def model_forward_gen(vision_model: bool = False):
    def model_forward(
        model,
        input_ids,
        attention_mask,
        position_ids,
        multi_modal_inputs: dict,
        logits_processor=None,
        logits_processor_args: dict = None,
        value_model=False,
        data_format: str = "thd",
        mtp_config: MtpConfig = None,
    ):
        """Forward pass for models with sequence packing."""
        assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"
        pre_process = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process = unwrap_model(model).post_process
        sp = unwrap_model(model).config.sequence_parallel
        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        mtp_enable_train = mtp_config and mtp_config.enable_train

        if data_format == "thd":
            input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=pre_process or (post_process and mtp_enable_train),
                use_fp8_padding=use_fp8_padding,
            )
            input_ids_rmpad = input_ids_rmpad.contiguous()

            # when pp > 1 and processor is not None, we need to pass the labels and loss_mask to the model
            if mtp_enable_train and post_process:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                model_kwargs["labels"] = args["label"].contiguous()
                model_kwargs["loss_mask"] = args["label_mask"].contiguous()

            input_args = dict(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
                packed_seq_params=packed_seq_params,
                **model_kwargs,
            )

            if vision_model:
                # workaround for supporting sequence packing with context parallelism
                # cp split with sequence packing will make model lose vision token information, so we need to keep
                # the original input_ids and pack them after vision embedding is calculated,
                # cooporate with mbridge
                input_args["input_ids"] = input_ids
                input_args["attention_mask"] = attention_mask

            output_orig = model(**input_args)

            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_packed_seqs(
                    output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
        elif data_format == "bshd":
            """
            data_format: "thd" or "bshd", default is "thd",
            why we need this?
                for some new models, GPT-OSS, the thd format is not supported, so we need to use the bshd format.
            When using the bshd format, we have to add paddings to the input_ids to meet the longest sequence length, 
            so it is recommended to disable dynamic batch size and set batch size to 1
            """
            assert fp8 is None, "fp8 is not supported for bshd format yet"

            batch_size, sequence_length = attention_mask.shape[:2]
            position_ids_for_preprocess = (
                torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                if vision_model
                else position_ids
            )
            pre_process_for_bshd = True if vision_model else pre_process
            new_input_ids, new_attention_mask, new_position_ids = preprocess_bshd(
                input_ids,
                attention_mask,
                position_ids_for_preprocess,
                sequence_parallel=sp,
                pre_process=pre_process_for_bshd,
            )
            output_orig = model(
                input_ids=new_input_ids,
                position_ids=None if vision_model else new_position_ids,
                attention_mask=new_attention_mask,
                **model_kwargs,
            )
            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_bshd(
                        v, attention_mask, position_ids_for_preprocess, sequence_parallel=sp, pre_process=True
                    )[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_bshd(
                        v, new_attention_mask, attention_mask, sequence_length, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_bshd(
                    output_orig, new_attention_mask, attention_mask, sequence_length, post_process=post_process
                )
        if value_model and post_process:
            output = output[..., 0]
        return output

    return model_forward


def _convert_to_nested_tensor(v, input_ids_lengths):
    """Convert regular tensor to NestedTensor, slicing according to input_ids_lengths.

    Args:
        v: Tensor to convert, shape [batch, seq_len]
        input_ids_lengths: List of valid lengths for each sample

    Returns:
        Converted NestedTensor
    """
    if isinstance(v, NestedTensor):
        return v

    batch_size = v.shape[0]
    assert len(input_ids_lengths) == batch_size, (
        f"len(input_ids_lengths)={len(input_ids_lengths)} != batch_size={batch_size}"
    )

    v_split_list = []
    for i in range(batch_size):
        vi = v[i]
        target_len = input_ids_lengths[i]
        if vi.shape[0] > target_len:
            vi = vi[:target_len]
        elif vi.shape[0] < target_len:
            vi = torch.cat([vi, torch.ones(target_len - vi.shape[0], dtype=vi.dtype, device=vi.device)])
        v_split_list.append(vi)

    v = torch.nested.nested_tensor(v_split_list, layout=torch.jagged)
    return v


def gptmodel_forward_model_engine(
    model,
    input_ids,
    multi_modal_inputs: dict,
    logits_processor=None,
    logits_processor_args: dict = None,
    value_model=False,
    vision_model=False,
    pad_token_id=None,
    data_format: str = "thd",
    mtp_enable_train: bool = False,
    local_cp_size: Optional[int] = None,
):
    """Default forward pass for GPT models with optional sequence packing."""

    assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process

    fp8 = unwrap_model(model).config.fp8
    use_fp8_padding = fp8 in ["e4m3", "hybrid"]

    model_kwargs = {}
    if "pixel_values" in multi_modal_inputs:
        model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
    if "image_grid_thw" in multi_modal_inputs:
        model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
    if "pixel_values_videos" in multi_modal_inputs:
        model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
    if "video_grid_thw" in multi_modal_inputs:
        model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

    batch_size = input_ids.shape[0]

    if data_format == "thd":
        use_prefix_tree = (logits_processor_args or {}).get("use_prefix_tree", False)
        prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")
        pb = _build_prefix_tree_batch(
            model, input_ids, logits_processor_args, use_prefix_tree,
            vision_model, mtp_enable_train,
        )
        if pb is not None:
            output = _forward_prefix_tree(
                model, pb, prefix_tree_attention, logits_processor,
                logits_processor_args, post_process, model_kwargs,
            )
        else:
            input_ids_rmpad, packed_seq_params, position_ids_rmpad = preprocess_thd_engine(
                input_ids,
                pre_process=pre_process or (post_process and mtp_enable_train),
                use_fp8_padding=use_fp8_padding,
                local_cp_size=local_cp_size,
            )
            input_ids_rmpad = input_ids_rmpad.contiguous()

            args = {}
            if mtp_enable_train and post_process:
                input_ids_offsets = input_ids.offsets()
                input_ids_lengths = input_ids_offsets.diff().tolist()

                for k in ["label", "loss_mask"]:
                    v = logits_processor_args[k]
                    v = _convert_to_nested_tensor(v, input_ids_lengths)
                    logits_processor_args[k] = v
                    args[k] = preprocess_thd_engine(
                        v,
                        pre_process=True,
                        need_roll=True,
                        use_fp8_padding=use_fp8_padding,
                        local_cp_size=local_cp_size,
                    )[0]

                model_kwargs["labels"] = args["label"].contiguous()
                model_kwargs["loss_mask"] = args["loss_mask"].contiguous()

            for _k in ("loss_mask", "use_prefix_tree", "prefix_tree_attention", "prefix_segments_batch"):
                if logits_processor_args and _k in logits_processor_args:
                    logits_processor_args.pop(_k)

            attention_mask = None
            if vision_model:
                input_ids_rmpad, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id)

            output_orig = model(
                input_ids=input_ids_rmpad,
                attention_mask=attention_mask,
                position_ids=position_ids_rmpad if not vision_model else None,
                packed_seq_params=packed_seq_params,
                **model_kwargs,
            )

            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_thd_engine(
                        v,
                        pre_process=True,
                        need_roll=(k == "label"),
                        use_fp8_padding=use_fp8_padding,
                        local_cp_size=local_cp_size,
                    )[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_thd_engine(
                        v, packed_seq_params, input_ids, batch_size,
                        post_process=post_process, local_cp_size=local_cp_size,
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_thd_engine(
                    output_orig, packed_seq_params, input_ids, batch_size,
                    post_process=post_process, local_cp_size=local_cp_size,
                )
    else:
        """
        data_format: "thd" or "bshd", default is "thd",
        why we need this?
            for some new models, GPT-OSS, the thd format is not supported, so we need to use the bshd format.
        When using the bshd format, we have to add paddings to the input_ids to meet the longest sequence length, 
        so it is recommended to disable dynamic batch size and set batch size to 1
        """
        assert local_cp_size is None, "dynamic_CP is not supported for bshd format"

        input_ids_bshd, attention_mask_bshd, position_ids_bshd = preprocess_bshd_engine(
            input_ids, pre_process=pre_process or (post_process and mtp_enable_train), use_fp8_padding=use_fp8_padding
        )

        if mtp_enable_train and post_process:
            args = {}
            input_ids_offsets = input_ids.offsets()
            input_ids_lengths = input_ids_offsets.diff().tolist()

            for k in ["label", "loss_mask"]:
                v = logits_processor_args[k]
                v = _convert_to_nested_tensor(v, input_ids_lengths)
                logits_processor_args[k] = v
                args[k] = preprocess_bshd_engine(v, pre_process=True, need_roll=True, use_fp8_padding=use_fp8_padding)[0]
            model_kwargs["labels"] = args["label"].contiguous()
            model_kwargs["loss_mask"] = args["loss_mask"].contiguous()

        if logits_processor_args and "loss_mask" in logits_processor_args:
            logits_processor_args.pop("loss_mask")

        if vision_model:
            input_ids_bshd, attention_mask = build_vlm_attn_mask_bshd(input_ids, batch_size, pad_token_id)
        else:
            attention_mask = attention_mask_bshd

        output_orig = model(
            input_ids=input_ids_bshd,
            attention_mask=attention_mask,
            position_ids=None if vision_model else position_ids_bshd,
            **model_kwargs,
        )
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_bshd_engine(
                    v, pre_process=True, need_roll=(k == "label"), use_fp8_padding=use_fp8_padding
                )[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_bshd_engine(v, attention_mask_bshd, post_process=post_process)
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_bshd_engine(output_orig, attention_mask_bshd, post_process=post_process)

    if value_model and post_process:
        output = output.squeeze(-1)

    return output


def _build_prefix_tree_batch(model, input_ids, logits_processor_args, use_prefix_tree, vision_model, mtp_enable_train):
    """Build prefix-tree micro-batch. Returns PrefixTreeMagiBatch or None."""
    if not use_prefix_tree or vision_model or mtp_enable_train:
        return None

    from verl.utils.prefix_tree_magi import build_prefix_tree_micro_batch

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")
    prefix_tree_dynamic = (logits_processor_args or {}).get("prefix_tree_dynamic", False)
    loss_mask_nested = (logits_processor_args or {}).get("loss_mask", None)
    prefix_segments_batch = (logits_processor_args or {}).get("prefix_segments_batch", None)
    if prefix_segments_batch is not None:
        import numpy as _np
        if isinstance(prefix_segments_batch, _np.ndarray):
            prefix_segments_batch = prefix_segments_batch.tolist()
        elif hasattr(prefix_segments_batch, "__iter__") and not isinstance(prefix_segments_batch, list):
            prefix_segments_batch = [el.data if hasattr(el, "data") else el for el in prefix_segments_batch]

    from megatron.core import parallel_state as _mpu
    return build_prefix_tree_micro_batch(
        model, input_ids, loss_mask_nested,
        prefix_segments_batch=prefix_segments_batch,
        attention_type=prefix_tree_attention,
        tp_size=_mpu.get_tensor_model_parallel_world_size(),
        cp_size=_mpu.get_context_parallel_world_size(),
        dynamic_trie=prefix_tree_dynamic,
    )


def _forward_prefix_tree(model, pt_batch, prefix_tree_attention, logits_processor, logits_processor_args, post_process, model_kwargs):
    """Forward pass for prefix-tree batches using magi or flex attention."""
    from verl.utils.prefix_tree_magi import restore_flat_to_nested

    flat_input_ids = pt_batch.local_flat_input_ids.unsqueeze(0)
    flat_position_ids = pt_batch.local_flat_position_ids.unsqueeze(0)

    for _k in ("loss_mask", "use_prefix_tree", "prefix_tree_attention", "prefix_segments_batch"):
        if logits_processor_args and _k in logits_processor_args:
            logits_processor_args.pop(_k)

    if prefix_tree_attention == "magi":
        output_orig = model(
            input_ids=flat_input_ids, attention_mask=None,
            position_ids=flat_position_ids, packed_seq_params=None,
            magi_attention_key=pt_batch.magi_key, **model_kwargs,
        )
    else:
        output_orig = model(
            input_ids=flat_input_ids, attention_mask=None,
            position_ids=flat_position_ids, packed_seq_params=None,
            flex_attention_key=pt_batch.flex_key, **model_kwargs,
        )

    from verl.utils.device import get_torch_device
    get_torch_device().synchronize()

    if post_process and logits_processor is not None:
        real_tokens = pt_batch.real_tokens
        if output_orig.shape[0] == 1:
            output_orig = output_orig[:, :real_tokens]
        else:
            output_orig = output_orig[:real_tokens].permute(1, 0, 2)

        output_orig_thd = output_orig.squeeze(0).unsqueeze(1)
        flat_label = torch.roll(pt_batch.flat_input_ids[:real_tokens], shifts=-1, dims=0).unsqueeze(1)
        orig_args = logits_processor_args or {}
        total_tokens = flat_label.shape[0]
        if "temperature" in orig_args:
            t = orig_args["temperature"]
            if isinstance(t, torch.Tensor) and t.is_nested:
                scalar_t = t.values()[0].item()
            elif isinstance(t, torch.Tensor):
                scalar_t = t.flatten()[0].item()
            else:
                scalar_t = float(t)
            flat_t = torch.full((total_tokens, 1), scalar_t, dtype=torch.float32, device=flat_label.device)
        else:
            flat_t = torch.ones(total_tokens, 1, dtype=torch.float32, device=flat_label.device)
        flat_args = {
            k: v for k, v in orig_args.items()
            if k not in ("label", "temperature", "loss_mask", "use_prefix_tree", "prefix_segments_batch")
        }
        flat_args["label"] = flat_label
        flat_args["temperature"] = flat_t
        output_dict = logits_processor(output_orig_thd, **flat_args)
        if isinstance(output_dict, dict) and "log_probs" in output_dict:
            lp_flat = output_dict["log_probs"].reshape(-1)[:real_tokens]
            output_dict["log_probs"] = restore_flat_to_nested(lp_flat, pt_batch)
        return output_dict
    else:
        real_tokens = pt_batch.real_tokens
        out_stripped = (
            output_orig[:, :real_tokens].squeeze(0)
            if output_orig.shape[0] == 1
            else output_orig[:real_tokens].squeeze(1)
        )
        return restore_flat_to_nested(out_stripped, pt_batch)
