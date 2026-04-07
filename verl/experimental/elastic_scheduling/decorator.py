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

# ---------------------------------------------------------------------------
# Elastic DP dispatch strategy
# ---------------------------------------------------------------------------
# ``make_elastic_dp_dispatch_fn`` returns a dict{dispatch_fn, collect_fn} that
# supports a **dynamically changing** DP group, e.g. after an elastic rebuild.
#
# Key difference from ``dispatch_lazy_compute_data_proto`` (nd_compute path):
#
#   * nd_compute relies on ``_dispatch_info`` / ``_collect_info`` caches keyed
#     by *mesh_name*.  After a rebuild those caches must be manually patched
#     by the controller, and removed ranks still occupy slots in the mapping.
#
#   * elastic_dp reads ``worker_group._elastic_dp_active_ranks``, a plain
#     ``list[int]`` of the *currently active* global ranks.  The controller
#     updates this attribute immediately after ``rebuild_dp_group`` returns.
#
# Dispatch rule (per global_rank i):
#   * active rank  → receives its own data shard  (index in active_ranks list)
#   * inactive rank → receives ``None``             (worker returns None)
#
# Collect rule:
#   * only gather outputs from active ranks; filter out None entries.
#   * concat the surviving DataProto/TensorDict objects.
#
# This makes ``_dispatch_info`` cache management unnecessary for the elastic
# path and eliminates the "removed rank pollutes mapping" problem.
# ---------------------------------------------------------------------------


import os
from verl.utils.ray_utils import parallel_put
from verl.protocol import BatchData

from verl.single_controller.base.decorator import (
    DISPATCH_MODE_FN_REGISTRY,
    Dispatch,
    _concat_data_proto_or_future,
    _split_args_kwargs_data_proto,
)


def dispatch_elastic_dp_data_proto(worker_group, *args, **kwargs):
    """Dispatch DataProto shards only to active DP ranks.

    Active ranks are determined by ``worker_group._elastic_dp_active_ranks``,
    a ``list[int]`` of wg-local global ranks that are currently participating
    in the DP group.  Inactive ranks receive ``None`` for every argument.

    The batch is split by ``len(active_ranks)`` (the *effective* DP size)
    rather than ``world_size``, so the per-rank shard size is correct.

    Each shard is uploaded to the Ray object store via ``parallel_put`` before
    routing so that the caller holds ObjectRefs (not raw CUDA tensors/views).
    This prevents ``CUDA illegal memory access`` errors that can occur when
    tensor views are pickled and transmitted as Ray task arguments.
    """
    active_ranks: list[int] = getattr(worker_group, "_elastic_dp_active_ranks", None)
    if active_ranks is None:
        raise RuntimeError(
            "worker_group._elastic_dp_active_ranks is not set. "
            "Call ElasticTrainer.register_elastic_unit_ranks() or "
            "_update_elastic_dp_active_ranks() before dispatching."
        )

    dp_size = len(active_ranks)
    if dp_size == 0:
        raise RuntimeError("worker_group._elastic_dp_active_ranks is empty – no active DP rank.")

    active_set = set(active_ranks)
    # Build shard index: active_ranks[i] → shard i
    rank_to_shard: dict[int, int] = {r: i for i, r in enumerate(active_ranks)}

    # Split batch into dp_size shards (one per active rank).
    # Use BatchData-based splitting which handles both DataProto and TensorDict.
    # Auto-padding is NOT used here: ElasticTrainer guarantees via
    # _update_required_samples() that batch_size is always divisible by dp_size
    # (the current active DP count), so no padding is needed.
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    max_workers = max(1, min(dp_size, os.cpu_count() or dp_size))
    put_splitted_args = [parallel_put(list(sa), max_workers=max_workers) for sa in splitted_args]
    put_splitted_kwargs = {k: parallel_put(list(sv), max_workers=max_workers) for k, sv in splitted_kwargs.items()}

    all_args: list[list] = [[] for _ in range(len(put_splitted_args))]
    all_kwargs: dict[str, list] = {k: [] for k in put_splitted_kwargs}

    for global_rank in range(worker_group.world_size):
        if global_rank in active_set:
            shard_idx = rank_to_shard[global_rank]
            for j, psa in enumerate(put_splitted_args):
                all_args[j].append(psa[shard_idx])
            for k, psv in put_splitted_kwargs.items():
                all_kwargs[k].append(psv[shard_idx])
        else:
            # Inactive rank: send None for every argument
            for j in range(len(put_splitted_args)):
                all_args[j].append(None)
            for k in put_splitted_kwargs:
                all_kwargs[k].append(None)

    return tuple(all_args), all_kwargs


def collect_elastic_dp_data_proto(worker_group, output):
    """Collect and concat DataProto results from active DP ranks only.

    Inactive ranks return ``None``; these are filtered out before concat.
    """
    active_ranks: list[int] = getattr(worker_group, "_elastic_dp_active_ranks", None)
    if active_ranks is None:
        raise RuntimeError("worker_group._elastic_dp_active_ranks is not set.")

    active_set = set(active_ranks)

    active_outputs = []
    for global_rank, out in enumerate(output):
        if global_rank in active_set and out is not None:
            active_outputs.append(out)

    assert active_outputs, "collect_elastic_dp_dataproto: no active output to collect."
    assert BatchData(active_outputs).is_concatable(), (
        f"expecting concatable output, but got element type {type(active_outputs[0])}"
    )
    return _concat_data_proto_or_future(active_outputs)


def collect_elastic_dp_metric(worker_group, output):
    """Collect metric outputs (list) from active DP ranks only (no concat)."""
    active_ranks: list[int] = getattr(worker_group, "_elastic_dp_active_ranks", None)
    if active_ranks is None:
        raise RuntimeError("worker_group._elastic_dp_active_ranks is not set.")

    active_set = set(active_ranks)
    return [out for global_rank, out in enumerate(output) if global_rank in active_set and out is not None]


Dispatch.register("ELASTIC_DP_COMPUTE_PROTO")
Dispatch.register("ELASTIC_DP_COMPUTE_METRIC")

DISPATCH_MODE_FN_REGISTRY[Dispatch.ELASTIC_DP_COMPUTE_PROTO] = {
    "dispatch_fn": dispatch_elastic_dp_data_proto,
    "collect_fn": collect_elastic_dp_data_proto,
}

DISPATCH_MODE_FN_REGISTRY[Dispatch.ELASTIC_DP_COMPUTE_METRIC] = {
    "dispatch_fn": dispatch_elastic_dp_data_proto,
    "collect_fn": collect_elastic_dp_data_proto,
}
