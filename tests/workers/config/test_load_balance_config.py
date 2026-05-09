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
"""Unit tests for LoadBalanceConfig and SGLangRouterConfig.

All tests run without Ray, torch, or tensordict — the config module has no
heavy-framework imports at the module level.
"""

import importlib.util
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load verl.base_config and verl.workers.config.load_balance
# without triggering verl/__init__ (which pulls in ray/torch/tensordict).
# ---------------------------------------------------------------------------


def _load_file(dotted_name: str, path: pathlib.Path):
    """Load a single Python source file as a module, bypassing the package __init__."""
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ROOT = pathlib.Path(__file__).parents[3]  # repo root

# 1. verl.base_config
_load_file("verl.base_config", _ROOT / "verl" / "base_config.py")

# 2. verl.workers.config.load_balance
_load_file("verl.workers.config.load_balance", _ROOT / "verl" / "workers" / "config" / "load_balance.py")

from verl.workers.config.load_balance import LoadBalanceConfig, SGLangRouterConfig  # noqa: E402

# ---------------------------------------------------------------------------
# SGLangRouterConfig
# ---------------------------------------------------------------------------


class TestSGLangRouterConfigDefaults:
    def test_default_policy(self):
        cfg = SGLangRouterConfig()
        assert cfg.policy == "cache_aware"

    def test_default_decode_policy(self):
        cfg = SGLangRouterConfig()
        assert cfg.decode_policy == "power_of_two"

    def test_default_cache_threshold(self):
        cfg = SGLangRouterConfig()
        assert cfg.cache_threshold == 0.3

    def test_default_balance_abs_threshold(self):
        cfg = SGLangRouterConfig()
        assert cfg.balance_abs_threshold == 64

    def test_default_balance_rel_threshold(self):
        cfg = SGLangRouterConfig()
        assert cfg.balance_rel_threshold == 1.5

    def test_default_health_check_interval(self):
        cfg = SGLangRouterConfig()
        assert cfg.health_check_interval_secs == 60


class TestSGLangRouterConfigValidPolicies:
    @pytest.mark.parametrize("policy", ["cache_aware", "round_robin", "consistent_hashing", "prefix_hash"])
    def test_valid_policy(self, policy):
        cfg = SGLangRouterConfig(policy=policy)
        assert cfg.policy == policy

    @pytest.mark.parametrize("decode_policy", ["power_of_two", "round_robin", "random"])
    def test_valid_decode_policy(self, decode_policy):
        cfg = SGLangRouterConfig(decode_policy=decode_policy)
        assert cfg.decode_policy == decode_policy


class TestSGLangRouterConfigInvalidPolicies:
    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="policy="):
            SGLangRouterConfig(policy="nonexistent")

    def test_invalid_decode_policy_raises(self):
        with pytest.raises(ValueError, match="decode_policy="):
            SGLangRouterConfig(decode_policy="cache_aware")  # valid router but not a decode policy

    def test_empty_policy_raises(self):
        with pytest.raises(ValueError):
            SGLangRouterConfig(policy="")


class TestSGLangRouterConfigThresholdValidation:
    def test_cache_threshold_zero(self):
        cfg = SGLangRouterConfig(cache_threshold=0.0)
        assert cfg.cache_threshold == 0.0

    def test_cache_threshold_one(self):
        cfg = SGLangRouterConfig(cache_threshold=1.0)
        assert cfg.cache_threshold == 1.0

    def test_cache_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="cache_threshold"):
            SGLangRouterConfig(cache_threshold=1.1)

    def test_cache_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="cache_threshold"):
            SGLangRouterConfig(cache_threshold=-0.1)

    def test_balance_abs_threshold_zero(self):
        cfg = SGLangRouterConfig(balance_abs_threshold=0)
        assert cfg.balance_abs_threshold == 0

    def test_balance_abs_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="balance_abs_threshold"):
            SGLangRouterConfig(balance_abs_threshold=-1)

    def test_balance_rel_threshold_exactly_one(self):
        cfg = SGLangRouterConfig(balance_rel_threshold=1.0)
        assert cfg.balance_rel_threshold == 1.0

    def test_balance_rel_threshold_below_one_raises(self):
        with pytest.raises(ValueError, match="balance_rel_threshold"):
            SGLangRouterConfig(balance_rel_threshold=0.9)

    def test_health_check_interval_zero_raises(self):
        with pytest.raises(ValueError, match="health_check_interval_secs"):
            SGLangRouterConfig(health_check_interval_secs=0)

    def test_health_check_interval_negative_raises(self):
        with pytest.raises(ValueError, match="health_check_interval_secs"):
            SGLangRouterConfig(health_check_interval_secs=-5)


class TestSGLangRouterConfigCustomValues:
    def test_custom_all_fields(self):
        cfg = SGLangRouterConfig(
            policy="round_robin",
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=2.0,
            health_check_interval_secs=30,
            decode_policy="random",
        )
        assert cfg.policy == "round_robin"
        assert cfg.cache_threshold == 0.5
        assert cfg.balance_abs_threshold == 32
        assert cfg.balance_rel_threshold == 2.0
        assert cfg.health_check_interval_secs == 30
        assert cfg.decode_policy == "random"


# ---------------------------------------------------------------------------
# LoadBalanceConfig
# ---------------------------------------------------------------------------


class TestLoadBalanceConfigDefaults:
    def test_default_strategy(self):
        cfg = LoadBalanceConfig()
        assert cfg.strategy == "least_inflight"

    def test_default_router_is_sglang_router_config(self):
        cfg = LoadBalanceConfig()
        assert isinstance(cfg.router, SGLangRouterConfig)

    def test_default_router_has_default_policy(self):
        cfg = LoadBalanceConfig()
        assert cfg.router.policy == "cache_aware"


class TestLoadBalanceConfigValidStrategies:
    @pytest.mark.parametrize("strategy", ["least_inflight", "sglang_router"])
    def test_valid_strategy(self, strategy):
        cfg = LoadBalanceConfig(strategy=strategy)
        assert cfg.strategy == strategy


class TestLoadBalanceConfigInvalidStrategy:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy="):
            LoadBalanceConfig(strategy="round_robin")  # valid for router, not for lb

    def test_empty_strategy_raises(self):
        with pytest.raises(ValueError):
            LoadBalanceConfig(strategy="")


class TestLoadBalanceConfigNestedRouterCoercion:
    def test_router_from_dict(self):
        cfg = LoadBalanceConfig(
            strategy="sglang_router",
            router={"policy": "round_robin"},
        )
        assert isinstance(cfg.router, SGLangRouterConfig)
        assert cfg.router.policy == "round_robin"

    def test_router_from_dict_with_all_fields(self):
        cfg = LoadBalanceConfig(
            strategy="sglang_router",
            router={
                "policy": "consistent_hashing",
                "cache_threshold": 0.5,
                "balance_abs_threshold": 32,
                "balance_rel_threshold": 2.0,
                "health_check_interval_secs": 30,
                "decode_policy": "random",
            },
        )
        assert cfg.router.policy == "consistent_hashing"
        assert cfg.router.cache_threshold == 0.5
        assert cfg.router.balance_abs_threshold == 32
        assert cfg.router.decode_policy == "random"

    def test_router_from_invalid_type_raises(self):
        with pytest.raises(TypeError, match="LoadBalanceConfig.router"):
            LoadBalanceConfig(strategy="sglang_router", router=42)

    def test_router_invalid_policy_in_dict_raises(self):
        with pytest.raises(ValueError, match="policy="):
            LoadBalanceConfig(strategy="sglang_router", router={"policy": "bad_policy"})


class TestLoadBalanceConfigMapping:
    """BaseConfig implements Mapping; verify dict-like access works."""

    def test_getitem_strategy(self):
        cfg = LoadBalanceConfig()
        assert cfg["strategy"] == "least_inflight"

    def test_get_with_default(self):
        cfg = LoadBalanceConfig()
        assert cfg.get("nonexistent", "fallback") == "fallback"

    def test_iter_yields_field_names(self):
        cfg = LoadBalanceConfig()
        keys = list(cfg)
        assert "strategy" in keys
        assert "router" in keys

    def test_len(self):
        cfg = LoadBalanceConfig()
        assert len(cfg) >= 2


# ---------------------------------------------------------------------------
# Integration: LoadBalanceConfig inside a minimal RolloutConfig-like dict
# (exercises the Hydra/OmegaConf coerce path in RolloutConfig.__post_init__)
# ---------------------------------------------------------------------------


class TestLoadBalanceConfigOmegaConfCoercion:
    def test_omegaconf_dict_config_coercion(self):
        """Simulate what Hydra/OmegaConf produces and confirm coercion works."""
        try:
            from omegaconf import OmegaConf

            raw = OmegaConf.create(
                {
                    "strategy": "sglang_router",
                    "router": {
                        "policy": "prefix_hash",
                        "cache_threshold": 0.4,
                        "balance_abs_threshold": 16,
                        "balance_rel_threshold": 1.2,
                        "health_check_interval_secs": 120,
                        "decode_policy": "round_robin",
                    },
                }
            )
            cfg = LoadBalanceConfig(**OmegaConf.to_container(raw, resolve=True))
            assert isinstance(cfg, LoadBalanceConfig)
            assert isinstance(cfg.router, SGLangRouterConfig)
            assert cfg.router.policy == "prefix_hash"
            assert cfg.router.decode_policy == "round_robin"
        except ImportError:
            pytest.skip("omegaconf not available")
