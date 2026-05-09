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
from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = ["SGLangRouterConfig", "LoadBalanceConfig"]

_VALID_STRATEGIES = frozenset({"least_inflight", "sglang_router"})
_VALID_ROUTER_POLICIES = frozenset({"cache_aware", "round_robin", "consistent_hashing", "prefix_hash"})
_VALID_DECODE_POLICIES = frozenset({"power_of_two", "round_robin", "random"})


@dataclass
class SGLangRouterConfig(BaseConfig):
    """Configuration for the sgl-model-gateway Rust Router.

    Only used when LoadBalanceConfig.strategy == "sglang_router".
    """

    # Routing policy for non-PD mode (also the prefill policy in PD mode).
    # Options: cache_aware, round_robin, consistent_hashing, prefix_hash
    policy: str = "cache_aware"

    # cache_aware: route to a replica when its prefix-match rate exceeds this value.
    cache_threshold: float = 0.3

    # cache_aware: switch to load-balance when (max_load - min_load) > this value.
    balance_abs_threshold: int = 64

    # cache_aware: switch to load-balance when max_load > min_load * this value.
    balance_rel_threshold: float = 1.5

    # How often the Router pings each worker's /health endpoint (seconds).
    health_check_interval_secs: int = 60

    # PD disaggregation: routing policy for the decode pool.
    # power_of_two is the recommended default (pick-best-of-2 by in-flight count).
    # Options: power_of_two, round_robin, random
    decode_policy: str = "power_of_two"

    def __post_init__(self) -> None:
        if self.policy not in _VALID_ROUTER_POLICIES:
            raise ValueError(
                f"SGLangRouterConfig.policy={self.policy!r} is not valid. Options: {sorted(_VALID_ROUTER_POLICIES)}"
            )
        if self.decode_policy not in _VALID_DECODE_POLICIES:
            raise ValueError(
                f"SGLangRouterConfig.decode_policy={self.decode_policy!r} is not valid. "
                f"Options: {sorted(_VALID_DECODE_POLICIES)}"
            )
        if not 0.0 <= self.cache_threshold <= 1.0:
            raise ValueError(f"SGLangRouterConfig.cache_threshold must be in [0, 1], got {self.cache_threshold}")
        if self.balance_abs_threshold < 0:
            raise ValueError(f"SGLangRouterConfig.balance_abs_threshold must be >= 0, got {self.balance_abs_threshold}")
        if self.balance_rel_threshold < 1.0:
            raise ValueError(
                f"SGLangRouterConfig.balance_rel_threshold must be >= 1.0, got {self.balance_rel_threshold}"
            )
        if self.health_check_interval_secs <= 0:
            raise ValueError(
                f"SGLangRouterConfig.health_check_interval_secs must be > 0, got {self.health_check_interval_secs}"
            )


@dataclass
class LoadBalanceConfig(BaseConfig):
    """Load balancing configuration for multi-replica rollout servers.

    Only effective when rollout.name=sglang and multiple replicas are deployed
    (data_parallel_size > 1 or multiple standalone replicas).
    """

    # Load balancing strategy.
    # - least_inflight: default; least in-flight requests (current behaviour, no extra deps)
    # - sglang_router:  delegate to sgl-model-gateway Rust Router (requires sglang_router pkg)
    strategy: str = "least_inflight"

    # sglang_router strategy settings; only used when strategy == "sglang_router".
    router: SGLangRouterConfig = field(default_factory=SGLangRouterConfig)

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"LoadBalanceConfig.strategy={self.strategy!r} is not valid. Options: {sorted(_VALID_STRATEGIES)}"
            )
        # Coerce nested config from dict / DictConfig (Hydra passes dicts)
        if isinstance(self.router, dict):
            object.__setattr__(self, "router", SGLangRouterConfig(**self.router))
        elif not isinstance(self.router, SGLangRouterConfig):
            try:
                from omegaconf import OmegaConf

                router_dict = OmegaConf.to_container(self.router, resolve=True)
            except Exception as exc:
                raise TypeError(
                    f"LoadBalanceConfig.router must be dict, DictConfig, or SGLangRouterConfig; "
                    f"got {type(self.router).__name__}."
                ) from exc
            object.__setattr__(self, "router", SGLangRouterConfig(**router_dict))
