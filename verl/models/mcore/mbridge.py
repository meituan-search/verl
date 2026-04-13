# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# VANILLA_MBRIDGE
try:
    from verl.models.mcore.patch import apply_patch_mbridge

    apply_patch_mbridge()
    from mbridge import AutoBridge as AB
    from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model
except ImportError:
    print("mbridge package not found. Please install mbridge with `pip install verl[mcore]` or `pip install mbridge`")
    raise

from collections import defaultdict
from typing import AsyncGenerator

import torch


class AutoBridge(AB):  # type: ignore
    @classmethod
    def from_config(cls, *args, **kwargs):
        bridge = super().from_config(*args, **kwargs)
        cls._inject_state_dict_loader(bridge)
        return bridge

    @classmethod
    def _inject_state_dict_loader(cls, bridge) -> None:
        bridge_cls = bridge.__class__
        if hasattr(bridge_cls, "load_weights_from_state_dict"):
            return

        # mbridge factories may return architecture-specific classes such as Qwen2Bridge.
        # Inject this API onto the concrete class so engine.set_param works for all of them.
        bridge_cls.load_weights_from_state_dict = cls.load_weights_from_state_dict
        bridge_cls.load_weights_from_async_generator = cls.load_weights_from_async_generator

    def load_weights_from_state_dict(
        self,
        models: list[torch.nn.Module],
        hf_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """
        Load weights from an in-memory HuggingFace state dict into a Megatron-Core model.

        This method has the same TP/EP scatter logic as load_weights(), but bypasses
        SafeTensorIO entirely — weights are read directly from ``hf_state_dict`` instead
        of from safetensors files on disk.

        Args:
            models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
            hf_state_dict: Mapping of HuggingFace parameter name → tensor.
                           Must contain every HF key that the model requires.
        """
        for i, model in enumerate(models):
            # map local weight names to global weight names
            local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
            # map local weight names to huggingface weight names
            local_to_hf_map = {
                k: self._weight_name_mapping_mcore_to_hf(v)
                for k, v in local_to_global_map.items()
                if "_extra_state" not in k
            }

            # determine which hf keys this rank is responsible for reading
            to_load = []
            for local_name, hf_names in local_to_hf_map.items():
                if ".mlp.experts.linear_fc" in local_name:
                    if self.mpu.etp_rank == 0:
                        to_load.extend(hf_names)
                else:
                    if self.mpu.tp_rank == 0:
                        to_load.extend(hf_names)
                    else:
                        # special case for lm_head.weight:
                        # every tp rank loads it when the model is a value model
                        if "lm_head.weight" in hf_names:
                            to_load.extend(hf_names)

            # validate that all required keys are present in the provided dict
            missing = set(to_load) - hf_state_dict.keys()
            if missing:
                raise KeyError(
                    f"load_weights_from_state_dict: the following HuggingFace keys are "
                    f"required but not found in hf_state_dict: {missing}"
                )

            # import mcore weights
            for local_name, hf_names in local_to_hf_map.items():
                param = model.state_dict()[local_name]

                # hf format → mcore format (only on the rank that owns the data)
                if set(to_load) & set(hf_names):
                    hf_weights = [hf_state_dict[x] for x in hf_names]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None

                # skip lm_head / embed_tokens for value models (shape[0] == 1)
                if hf_names[0] in {"lm_head.weight", "model.embed_tokens.weight"}:
                    if param.shape[0] == 1 and (mcore_weight is None or mcore_weight.shape[0] != 1):
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous() for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load)

    async def load_weights_from_async_generator(
        self,
        models: list[torch.nn.Module],
        hf_weight_generator: AsyncGenerator[tuple[str, torch.Tensor], None],
    ) -> None:
        """
        Load weights from a per-tensor generator into Megatron-Core models.
        """
        # Phase 1: build per-model mappings and a global reverse index.
        if not hasattr(self, "_generator_per_model_meta"):
            per_model_meta: list[tuple[dict, torch.nn.Module]] = []
            for model in models:
                local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
                local_to_hf_map = {
                    k: self._weight_name_mapping_mcore_to_hf(v)
                    for k, v in local_to_global_map.items()
                    if "_extra_state" not in k
                }
                per_model_meta.append((local_to_hf_map, model))

            hf_key_to_targets: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for model_idx, (local_to_hf_map, _) in enumerate(per_model_meta):
                for local_name, hf_names in local_to_hf_map.items():
                    for hf_key in hf_names:
                        hf_key_to_targets[hf_key].append((model_idx, local_name))

            self._generator_per_model_meta = per_model_meta
            self._generator_hf_key_to_targets = hf_key_to_targets
        else:
            per_model_meta = self._generator_per_model_meta
            hf_key_to_targets = self._generator_hf_key_to_targets

        # Phase 2: flush helper (identical to the sync version)
        def _flush(
            model_idx: int,
            local_name: str,
            hf_names: list[str],
            buffer: dict[str, torch.Tensor],
        ) -> None:
            """Convert buffered HF tensors → mcore format → slice own TP shard → copy."""
            _, model = per_model_meta[model_idx]
            param = model.state_dict()[local_name]

            hf_weights = [buffer[x] for x in hf_names]
            mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)

            # skip lm_head / embed_tokens for value models (shape[0] == 1)
            if hf_names[0] in {
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
            }:
                if param.shape[0] == 1 and mcore_weight.shape[0] != 1:
                    return

            # Each rank independently slices its own shard — no communication needed.
            if ".mlp.experts.linear_fc" in local_name:
                shards = list(self._weight_split_across_tp(local_name, mcore_weight, param, self.mpu.etp_size))
                shard = shards[self.mpu.etp_rank]
            else:
                shards = list(self._weight_split_across_tp(local_name, mcore_weight, param, self.mpu.tp_size))
                shard = shards[self.mpu.tp_rank]

            param.copy_(shard.to(param.device, dtype=param.dtype).contiguous())

        # Phase 3: stream the async generator, buffer multi-key params
        expected_hf_keys: set[str] = set(hf_key_to_targets.keys())
        received_hf_keys: set[str] = set()
        pending: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)

        async for hf_key, tensor in hf_weight_generator:
            if hf_key in expected_hf_keys:
                received_hf_keys.add(hf_key)

            targets = hf_key_to_targets.get(hf_key)
            if targets is None:
                continue  # key not needed by this model — discard immediately

            for model_idx, local_name in targets:
                local_to_hf_map = per_model_meta[model_idx][0]
                hf_names = local_to_hf_map[local_name]

                slot = pending[(model_idx, local_name)]
                slot[hf_key] = tensor.clone()

                # flush as soon as every HF key for this param has arrived
                if all(k in slot for k in hf_names):
                    _flush(model_idx, local_name, hf_names, slot)
                    del pending[(model_idx, local_name)]

        missing_keys = expected_hf_keys - received_hf_keys
        if missing_keys:
            raise KeyError(
                f"load_weights_from_async_generator: the following HuggingFace keys are "
                f"required but were never yielded by the generator: {missing_keys}"
            )


__all__ = ["AutoBridge", "make_value_model", "freeze_moe_router"]
