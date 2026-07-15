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

"""Prefix-tree helpers consumed by verl trainers (SFT, PPO).

Every public function here is a single call that checks *config* internally —
the caller never needs to gate on ``use_prefix_tree``.
"""

from __future__ import annotations

# ──────────────────────── engine config ──────────────────────────


def apply_engine_config(engine_config, config_or_data: dict) -> None:
    """Thread prefix-tree flags from config into *engine_config*."""
    engine_config.use_prefix_tree = config_or_data.get("use_prefix_tree", False)
    engine_config.prefix_tree_attention = config_or_data.get("prefix_tree_attention", "flex")
    engine_config.prefix_tree_olb_backend = config_or_data.get("prefix_tree_olb_backend", None)


# ──────────────────────── meta info ─────────────────────────────


def add_meta_info(meta_dict: dict, config_or_data: dict) -> None:
    """Add prefix-tree entries to a meta-info dict (mutates in-place)."""
    meta_dict["use_prefix_tree"] = config_or_data.get("use_prefix_tree", False)
    meta_dict["prefix_tree_attention"] = config_or_data.get("prefix_tree_attention", "flex")
    meta_dict["prefix_tree_olb_backend"] = config_or_data.get("prefix_tree_olb_backend", None)


# ──────────────────────── metrics ───────────────────────────────


def compute_metrics(metrics: dict, input_ids, config_or_data: dict) -> None:
    """Compute prefix-sharing metrics if *use_prefix_tree* is enabled.

    Updates *metrics* in-place with keys like ``prefix_tree/sharing_ratio``.
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return
    from verl.utils.prefix_tree.dynamic import compute_prefix_tree_metrics

    metrics.update(compute_prefix_tree_metrics(input_ids))


# ──────────────────────── disable for log-prob ───────────────────


def configure_olb_backend(batch_td, config_or_data: dict) -> None:
    """Apply *prefix_tree_olb_backend* override before old log-prob forward.

    Called before ``compute_log_prob`` — not on fallback.  The fallback
    (inside the try/except in ``_compute_old_log_prob``) always disables
    prefix-tree entirely regardless of config.

    - ``None``: no-op, keep training backend
    - ``"magi"`` / ``"flex"``: override to that prefix-tree backend
    - ``"fa3"``: explicitly disable prefix-tree (use plain FA3) for this pass
    - anything else: raises ``ValueError`` — prevents silent fallback to FA3
    """
    if not _is_prefix_tree_enabled(config_or_data):
        return
    from verl.utils import tensordict_utils as tu

    olb = config_or_data.get("prefix_tree_olb_backend", None)
    if olb is None:
        return
    if olb in ("magi", "flex"):
        tu.assign_non_tensor(batch_td, prefix_tree_attention=olb)
    elif olb == "fa3":
        tu.assign_non_tensor(batch_td, use_prefix_tree=False)
    else:
        raise ValueError(f"Unknown prefix_tree_olb_backend {olb!r}. Valid values: None, 'magi', 'flex', 'fa3'.")


# ──────────────────────── internal ──────────────────────────────


def _is_prefix_tree_enabled(config_or_data) -> bool:
    if isinstance(config_or_data, dict):
        return config_or_data.get("use_prefix_tree", False)
    return bool(getattr(config_or_data, "use_prefix_tree", False))
