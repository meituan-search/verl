# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""GPU-free unit tests for LoadBalanceConfig dataclass."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from verl.workers.config.rollout import LoadBalanceConfig


def test_load_balance_config_defaults():
    cfg = LoadBalanceConfig()
    assert cfg.capacity_threshold == 0.85
    assert cfg.poll_interval_ms == 200
    assert cfg.backend == "sglang"


def test_load_balance_config_from_yaml():
    yaml = """
    capacity_threshold: 0.90
    poll_interval_ms: 100
    backend: vllm
    """
    raw = OmegaConf.create(yaml)
    cfg = LoadBalanceConfig(**raw)
    assert cfg.capacity_threshold == 0.90
    assert cfg.backend == "vllm"
