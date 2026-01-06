# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""Utility functions for dumping quantized weights from SGLang engine."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def dump_flash_rl_quantized_weights(
    quantized_weights_list: List[Dict],
    quantized_scales: Optional[Dict] = None,
    dump_dir: Optional[str] = None,
    prefix: str = "quantized_weights_flash_rl",
) -> str:
    """Dump flash_rl量化后的权重到JSON文件（与nvfp8格式一致）
    
    Args:
        quantized_weights_list: List of dicts with keys: name, shape, dtype, max, min, mean, std, scale (optional stats dict)
        quantized_scales: Optional dict mapping parameter names to scales (deprecated, kept for compatibility)
        dump_dir: Directory to dump weights to. If None, uses default.
        prefix: Prefix for the dump file name.
    
    Returns:
        Path to the dumped file.
    """
    # Use flashrl subdirectory as default
    if dump_dir is None:
        base_dump_dir = "/workdir/quant-rollout/work_logs/weight_dumps"
        dump_dir = os.path.join(base_dump_dir, "flashrl")
    
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    print(f"[FLASH_RL DUMP] Dump directory: {dump_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = os.path.join(dump_dir, f"{prefix}_{timestamp}.json")
    print(f"[FLASH_RL DUMP] Dump file path: {dump_path}")
    
    dump_data = []
    
    for item in quantized_weights_list:
        name = item["name"]
        
        # Check if item already contains computed stats (new format)
        if "max" in item and "min" in item and "mean" in item and "std" in item:
            # New format: stats already computed
            stats = {
                "module_name": name,
                "shape": item["shape"],
                "dtype": item["dtype"],
                "max": item["max"],
                "min": item["min"],
                "mean": item["mean"],
                "std": item["std"],
            }
            
            # Add scale information if available
            if item.get("scale") is not None:
                stats["weight_scale_inv"] = item["scale"]
        else:
            # Old format: contains tensor, need to compute stats
            weight = item["weight"]
            scale = item.get("scale", None)
            
            # If scale is not in item, try to get from quantized_scales dict
            if scale is None and quantized_scales is not None:
                scale = quantized_scales.get(name)
            
            def compute_tensor_stats(tensor):
                """Compute statistics for a tensor, converting to FP32 if needed."""
                if tensor.dtype == torch.float8_e4m3fn:
                    tensor_fp32 = tensor.to(torch.float32)
                else:
                    tensor_fp32 = tensor.float()
                
                return {
                    "max": float(tensor_fp32.max().item()),
                    "min": float(tensor_fp32.min().item()),
                    "mean": float(tensor_fp32.mean().item()),
                    "std": float(tensor_fp32.std().item()),
                }
            
            # Compute weight statistics
            weight_stats = compute_tensor_stats(weight)
            
            stats = {
                "module_name": name,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                **weight_stats,
            }
            
            # Add scale information if available
            if scale is not None:
                if isinstance(scale, torch.Tensor):
                    scale_stats = compute_tensor_stats(scale)
                    stats["weight_scale_inv"] = {
                        "shape": list(scale.shape),
                        "dtype": str(scale.dtype),
                        **scale_stats,
                    }
                elif isinstance(scale, dict):
                    # Handle nested scale dictionary (for sharded weights)
                    scale_dict_stats = {}
                    for key, scale_tensor in scale.items():
                        if isinstance(scale_tensor, torch.Tensor):
                            scale_stats = compute_tensor_stats(scale_tensor)
                            scale_dict_stats[str(key)] = {
                                "shape": list(scale_tensor.shape),
                                "dtype": str(scale_tensor.dtype),
                                **scale_stats,
                            }
                    stats["weight_scale_inv"] = scale_dict_stats
        
        dump_data.append(stats)
    
    try:
        with open(dump_path, 'w') as f:
            json.dump(dump_data, f, indent=2)
        print(f"[FLASH_RL DUMP] ✓ Successfully wrote {len(dump_data)} weights to {dump_path}")
        logger.info(f"[QuantizedRL] Dumped {len(dump_data)} quantized weights to {dump_path}")
        return dump_path
    except Exception as e:
        print(f"[FLASH_RL DUMP] ✗ Failed to write dump file: {e}")
        raise

