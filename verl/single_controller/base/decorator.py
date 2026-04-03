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
import inspect
from functools import partial, wraps
from types import FunctionType

from verl.protocol import DataProtoFuture, _padding_size_key
from verl.utils.py_functional import DynamicEnum

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(DynamicEnum):
    """Enum class defining different dispatch modes for distributed computation.

    Each mode represents a specific strategy for distributing data across
    different ranks in a distributed system. The modes are used to control
    how data is partitioned and processed across different worker groups.
    """

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


class Execute(DynamicEnum):
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
init_predefined_dispatch_mode()
init_predefined_execute_mode()


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import BatchData

    splitted_args = []
    for arg in args:
        assert BatchData(arg).is_chunkable(), f"arg of type {type(arg)} is not chunkable"
        chunked_arg = BatchData(arg).chunk(chunks=chunks)
        assert len(chunked_arg) == chunks
        splitted_args.append(chunked_arg)

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert BatchData(val).is_chunkable(), f"kwarg '{key}' of type {type(val)} is not chunkable"
        chunked_kwarg = BatchData(val).chunk(chunks=chunks)
        assert len(chunked_kwarg) == chunks
        splitted_kwargs[key] = chunked_kwarg

    return splitted_args, splitted_kwargs


def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    data_proto_len = None
    padding_size = None

    def _padding_and_split_data(obj, chunks):
        nonlocal data_proto_len, padding_size
        assert isinstance(obj, DataProto | DataProtoFuture)
        if isinstance(obj, DataProto) and obj.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(obj)
                padding_size = (chunks - (data_proto_len % chunks)) if (data_proto_len % chunks > 0) else 0
            else:
                assert data_proto_len == len(obj), (
                    f"expecting all arg share same length of {data_proto_len}, but got {len(obj)}"
                )
            obj.padding(padding_size=padding_size)
        return obj.chunk(chunks=chunks)

    splitted_args = [_padding_and_split_data(arg, chunks) for arg in args]
    splitted_kwargs = {key: _padding_and_split_data(val, chunks) for key, val in kwargs.items()}
    if padding_size is not None:
        splitted_kwargs[_padding_size_key] = padding_size

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def _concat_data_proto_or_future(output: list):
    from verl.protocol import BatchData

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    return BatchData(output).concat()


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_nd_compute(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    import os

    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.utils.ray_utils import parallel_put

    assert isinstance(worker_group, WorkerGroup)

    max_workers = max(1, min(len(args[0]), os.cpu_count()))

    args = [parallel_put(arg, max_workers=max_workers) for arg in args]
    kwargs = {k: parallel_put(v, max_workers=max_workers) for k, v in kwargs.items()}

    all_args = []
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_nd_compute(collect_mask: list[bool], worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size

    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        collect_dp_rank = collect_mask[global_rank]
        if collect_dp_rank:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_nd_compute_dataproto(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    return dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *splitted_args, **splitted_kwargs)


def collect_nd_compute_dataproto(collect_mask: list[bool], worker_group, output):
    output = collect_nd_compute(collect_mask, worker_group, output)

    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )
    return _concat_data_proto_or_future(output)


def dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # query dispatch info of the worker group
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    # perform dispatch
    dp_size = max(dp_rank_mapping) + 1
    return dispatch_nd_compute_dataproto(dp_rank_mapping, dp_size, worker_group, *args, **kwargs)


def collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # the dispatch info is stored in the worker group
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    # a boolean of whether the dp_rank is used for collect
    collect_mask = worker_group._collect_info[mesh_name]
    # perform dispatch
    return collect_nd_compute_dataproto(collect_mask, worker_group, *args, **kwargs)


def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    return {
        "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
        "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
    }


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


def dispatch_elastic_dp_dataproto(worker_group, *args, **kwargs):
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
    import os

    from verl.utils.ray_utils import parallel_put

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

    # Upload active shards to Ray object store so that each rank receives an
    # ObjectRef rather than a raw tensor view.  This is critical for CUDA safety:
    # without parallel_put the shard (a view of the original tensor) would be
    # pickled on the driver and sent as a task argument, which can trigger
    # "CUDA illegal memory access" when the backing memory is reclaimed before
    # the remote task deserializes it.
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


def collect_elastic_dp_dataproto(worker_group, output):
    """Collect and concat DataProto results from active DP ranks only.

    Inactive ranks return ``None``; these are filtered out before concat.
    """
    active_ranks: list[int] = getattr(worker_group, "_elastic_dp_active_ranks", None)
    if active_ranks is None:
        raise RuntimeError("worker_group._elastic_dp_active_ranks is not set.")

    active_set = set(active_ranks)

    from verl.protocol import BatchData

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


def make_elastic_dp_dispatch_fn():
    """Return dispatch/collect fn pair for elastic DP DataProto methods.

    The returned dict is suitable as the ``dispatch_mode`` argument to
    ``@register``, e.g.::

        @register(dispatch_mode=make_elastic_dp_dispatch_fn())
        def update_actor(self, data): ...

    The controller must set ``worker_group._elastic_dp_active_ranks`` to a
    ``list[int]`` of the currently active global ranks *before* calling any
    method decorated with this dispatch mode.  ElasticTrainer does this
    automatically in ``_update_elastic_dp_active_ranks()`` after each
    ``rebuild_dp_group`` call.
    """
    return {
        "dispatch_fn": dispatch_elastic_dp_dataproto,
        "collect_fn": collect_elastic_dp_dataproto,
    }


def make_elastic_dp_metric_dispatch_fn():
    """Like ``make_elastic_dp_dispatch_fn`` but collects raw metric lists."""
    return {
        "dispatch_fn": dispatch_elastic_dp_dataproto,
        "collect_fn": collect_elastic_dp_metric,
    }


# Global registry for dispatch mode.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(dispatch_mode, Dispatch | dict), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.

    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """

    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
