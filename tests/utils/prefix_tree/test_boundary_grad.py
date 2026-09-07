# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""Grad-flow test for the fused MAGI+CP boundary gather (reviewer finding + 4x follow-up).

post_processing_packed_lce reassembles boundary log-probs across CP ranks with
a collective all_gather. Two failure modes, both found in the wild:
  (a) plain all_gather detaches: forward VALUES are right but the actor update
      backprops NOTHING through boundary tokens (original reviewer finding);
  (b) an autograd-aware all_gather whose backward all_reduces: under static CP
      the restore+loss is REPLICATED on every CP rank, so every rank's
      consumption reaches the owner and the boundary gradient is counted
      CP-world-fold (the 4x tail-row amplification found by the per-token
      grad probe).

Correct semantics: the owner rank patches with its own locally-produced tail
log-prob (already in its autograd graph — flows exactly once); other ranks
patch with detached gathered VALUES. This test runs the real collective path
on a 2-process gloo group and asserts the end-to-end property: after a
replicated backward on every rank, each entry's gradient reaches its producer
EXACTLY ONCE (grad == 1.0, not world_size).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import socket
import sys
import types


def _install_megatron_stubs() -> None:
    # Spawned children don't run conftest: install the same stubs BEFORE
    # importing forward (mirrors tests/utils/prefix_tree/conftest.py).
    try:
        import megatron  # noqa: F401

        return
    except Exception:
        pass

    class _StubModule(types.ModuleType):
        def __getattr__(self, item):
            if item.startswith("__"):
                raise AttributeError(item)
            child = _make_stub(f"{self.__name__}.{item}")
            setattr(self, item, child)
            sys.modules[child.__name__] = child
            return child

    def _make_stub(name):
        mod = _StubModule(name)
        mod.__path__ = []
        mod.__package__ = name
        mod.__file__ = f"<stub:{name}>"
        mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        return mod

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        _prefixes = ("megatron", "magi_attention", "apex", "transformer_engine")

        def find_spec(self, fullname, path=None, target=None):
            for prefix in self._prefixes:
                if fullname == prefix or fullname.startswith(prefix + "."):
                    return importlib.util.spec_from_loader(fullname, loader=self)
            return None

        def create_module(self, spec):
            return _make_stub(spec.name)

        def exec_module(self, module):
            pass

    for pkg in _StubFinder._prefixes:
        sys.modules.setdefault(pkg, _make_stub(pkg))
    sys.meta_path.insert(0, _StubFinder())


_install_megatron_stubs()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from verl.utils.prefix_tree import forward as fwd  # noqa: E402
from verl.utils.prefix_tree.dynamic import build_tree_dynamic  # noqa: E402
from verl.utils.prefix_tree.magi import PackRestorationParam, PrefixTreeMagiBatch  # noqa: E402
from verl.utils.prefix_tree.utils import build_layout_from_tree_node  # noqa: E402

_SAMPLES = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 3, 6])]


def _build_pb() -> PrefixTreeMagiBatch:
    subtrie = build_tree_dynamic(_SAMPLES)
    params = build_layout_from_tree_node(_SAMPLES, subtrie)
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
            boundary_registry=getattr(params, "boundary_registry", None),
        ),
        subtrie=subtrie,
        real_tokens=params.tree_packed_tokens.shape[0],
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, port, mode):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)

    # Point forward's mpu at our real gloo group as the CP group.
    orig = (fwd.mpu.get_context_parallel_world_size, fwd.mpu.get_context_parallel_group)
    fwd.mpu.get_context_parallel_world_size = lambda: world_size
    fwd.mpu.get_context_parallel_group = lambda: dist.group.WORLD
    try:
        pb = _build_pb()
        registry = pb.restoration.boundary_registry
        assert registry, "expected a boundary registry from the 3-way fork"
        boundary_pos, leaves = registry[0]
        assert len(leaves) == 3, f"expected 3 leaves at the fork, got {len(leaves)}"

        # Split the leaves across CP ranks; each rank produces only its own
        # boundary values (grad-carrying), as the LCE pass would.
        if mode == "single_rank":
            owned = list(range(3)) if rank == 0 else []
        else:
            owned = [s for s in range(3) if s % world_size == rank]
        srcs = {s: torch.tensor(100.0 * rank + s, requires_grad=True) for s in owned}
        pb._boundary_local_vals = [(boundary_pos, s, srcs[s]) for s in owned]

        fwd.post_processing_packed_lce(pb, magi_key=object())

        # Coverage + values: every leaf's boundary log-prob arrived with its own value.
        assert set(pb._boundary_logps) == {0, 1, 2}, f"rank{rank}: bad coverage {sorted(pb._boundary_logps)}"
        for s, vals in pb._boundary_logps.items():
            ((pos, v),) = vals
            owner = 0 if mode == "single_rank" else s % world_size
            assert pos == boundary_pos
            assert abs(v.item() - (100.0 * owner + s)) < 1e-4, f"rank{rank} sample{s}: wrong value {v.item()}"
            if owner == rank:
                # The producer's own copy must stay graph-connected — the
                # original all_gather-detach bug cut even this. (requires_grad,
                # not grad_fn: the producer's value may be a leaf tensor.)
                assert v.requires_grad, (
                    f"rank{rank} sample{s}: OWN boundary log-prob is DETACHED — "
                    f"the cross-CP gather cut the autograd graph"
                )

        # Replicated backward: every rank consumes ALL boundary values (static-CP
        # replicas run the same restore+loss). Each producer must receive the
        # gradient EXACTLY ONCE — grad 1.0, not world_size (the 4x overcount).
        # (A rank owning no boundaries has only detached copies: skip backward.)
        all_vals = [v for vals in pb._boundary_logps.values() for _, v in vals]
        if any(v.requires_grad for v in all_vals):
            torch.stack(all_vals).sum().backward()
        for s, t in srcs.items():
            assert t.grad is not None, f"rank{rank}: sample{s} source got no grad at all"
            assert abs(t.grad.item() - 1.0) < 1e-6, (
                f"rank{rank}: sample{s} grad is {t.grad.item()} — expected exactly 1.0; "
                f"a world_size-fold value means every CP replica's consumption was counted"
            )
    finally:
        fwd.mpu.get_context_parallel_world_size, fwd.mpu.get_context_parallel_group = orig
        dist.destroy_process_group()


def _run(mode: str) -> None:
    port = _free_port()
    mp.spawn(_worker, args=(2, port, mode), nprocs=2, join=True)


def test_boundary_grad_flow_interleaved_ranks():
    """Rank0 owns leaves {0,2}, rank1 owns {1}: every gathered value must keep its graph."""
    _run("interleaved")


def test_boundary_grad_flow_single_rank_owner():
    """All boundaries on rank0, rank1 contributes none (zero-entry padding path):
    rank1's copies must still be grad-connected so its downstream usage backprops."""
    _run("single_rank")
