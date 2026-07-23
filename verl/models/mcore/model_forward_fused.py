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

from collections import OrderedDict
from typing import Optional

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from packaging import version
from torch import Tensor

from verl.models.mcore.util import preprocess_packed_seqs, preprocess_thd_engine
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs_for_dict_output, postprocess_thd_engine


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    assert version.parse(mcore.__version__) >= version.parse("0.13.0"), (
        "Fused forward patching requires mecore >= 0.13.0"
    )
    model = _get_patching_model(model)
    if model is not None:
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model.forward = model.forward_backup


def fused_forward_model_gen(vision_model: bool = False):
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
    ):
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process: bool = unwrap_model(model).post_process

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
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
        labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        input_args = dict(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if vision_model:
            # workaround for supporting sequence packing with context parallelism
            # cp split with sequence packing will make model lose vision token information, so we need to keep
            # the original input_ids and pack them after vision embedding is calculated,
            # cooporate with mbridge
            input_args["input_ids"] = input_ids
            input_args["attention_mask"] = attention_mask

        output_orig: CausalLMOutputForPPO = model(**input_args)

        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def fused_forward_model_engine(vision_model: bool = False):
    def fused_forward_model_engine_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
        logits_processor_args: Optional[dict] = None,
    ):
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process

        # Prefix-tree (MAGI) fused path: all PP stages use MAGI-dispatched tokens.
        # - pre_process=True (stage 0): dispatch → embedding → MAGI → raw hidden out
        # - pre_process=False, post_process=False (intermediate): recv CP-local →
        #   MAGI → raw hidden out
        # - post_process=True (last stage): recv CP-local → MAGI → undispatch → LCE
        _pt_args = logits_processor_args or {}
        use_prefix_tree = _pt_args.get("use_prefix_tree", False)

        if use_prefix_tree:
            # Rolling (copied from preprocess_thd_engine) is needed before pack;
            # FP8/CP alignment skipped — handled by prefix-tree's own path.
            # Guard defers VLM-with-images: 3D M-RoPE position handling not yet wired
            # (prefix_tree_rope_context assumes 1D). Text-only on ViT-config models passes through.
            from verl.utils.prefix_tree.forward import fuse_try_forward_prefix_tree

            output = fuse_try_forward_prefix_tree(
                model=model,
                input_ids=input_ids,
                labels=labels,
                temperature=temperature,
                logits_processor_args=_pt_args,
                calculate_entropy=calculate_entropy,
                vision_model=vision_model,
                has_vision_data="pixel_values" in multi_modal_inputs,
            )
            if output is not None:
                return output

        # Standard FA3 fused path (unchanged from upstream):
        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]

        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids, pre_process=pre_process, use_fp8_padding=use_fp8_padding
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        attention_mask = None
        if vision_model:
            input_ids_rmpad = input_ids.to_padded_tensor(pad_token_id)
            seqlens_in_batch = input_ids.offsets().diff().to(input_ids.device)
            max_seq_len = input_ids_rmpad.shape[1]
            attention_mask = torch.arange(max_seq_len, device=input_ids.device).unsqueeze(
                0
            ) < seqlens_in_batch.unsqueeze(1)

        labels_rmpad, _, _ = preprocess_thd_engine(
            labels, pre_process=True, need_roll=True, use_fp8_padding=use_fp8_padding
        )
        labels_rmpad = labels_rmpad.contiguous()

        output_orig: CausalLMOutputForPPO = model(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = postprocess_thd_engine(
            log_probs, packed_seq_params, input_ids, input_ids.shape[0], post_process=post_process
        )

        output = {"log_probs": log_probs}

        if calculate_entropy:
            entropy = output_orig.entropy
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(0)
            entropy = postprocess_thd_engine(
                entropy, packed_seq_params, input_ids, input_ids.shape[0], post_process=post_process
            )
            output["entropy"] = entropy

        return output

    return fused_forward_model_engine_inner


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?

    Handles both the standard fused path and the fused prefix-tree path:
    - Prefix-tree (``magi_attention_key`` + ``pt_batch`` in kwargs): installs
      the rope override and decoder-key plumbing, then delegates to
      ``fuse_forward_body`` (magi.py).
    - Standard fused (no attention key): runs preprocess → decoder → LCE.
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    # Prefix-tree fused path: pop keys only when pt_batch is present.
    # Unfused path leaves keys in **kwargs for the patched TransformerBlock.
    pt_batch = kwargs.pop("pt_batch", None)
    if pt_batch is not None:
        _magi_key = kwargs.pop("magi_attention_key", None)
        _flex_key = kwargs.pop("flex_attention_key", None)
        if _magi_key is not None or _flex_key is not None:
            from verl.utils.prefix_tree.forward import fuse_forward_body
            from verl.utils.prefix_tree.magi import (
                prefix_tree_decoder_key_context,
                prefix_tree_rope_context,
            )

            with (
                prefix_tree_rope_context(model, position_ids),
                prefix_tree_decoder_key_context(model, _magi_key, _flex_key),
            ):
                return fuse_forward_body(
                    model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    temperature=temperature,
                    pt_batch=pt_batch,
                    magi_key=_magi_key,
                    flex_key=_flex_key,
                    decoder_input=decoder_input,
                    packed_seq_params=packed_seq_params,
                    extra_block_kwargs=extra_block_kwargs,
                    inference_context=inference_context,
                )

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    # Run decoder.
    hidden_states = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
        **kwargs,
    )

    if not model.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # Get the output weight - use embedding weight if output_layer is None or weight is shared
    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        # When embeddings are tied, use the embedding weight
        output_weight = model.embedding.word_embeddings.weight

    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        output_weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
