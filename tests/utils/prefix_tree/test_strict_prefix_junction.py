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

"""Strict-prefix junction coverage for the fused LCE boundary registry.

A junction where a sample terminates (its whole sequence is a strict token
prefix of another sample's) must be registered like a fork: the continuing
sample reads the junction position, so it needs its own next-token label.
Pre-fix, only >=2-children nodes were registered, so the continuing sample's
boundary log-prob came from the terminating owner's rolled 0-pad label —
silently wrong. Triggers on multi-turn/agentic data, not on ordinary
shared-prompt forks or GRPO n-siblings.
"""

from __future__ import annotations

import torch

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import PackRestorationParam, PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _build_pb(samples, subtrie, params):
    """Mirror _finalize_prefix_tree_batch's PrefixTreeMagiBatch wrapping."""
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


def _layout(samples):
    subtrie = build_tree_dynamic(samples)
    assert subtrie is not None, "samples share a prefix, a subtrie must exist"
    params = build_layout_from_tree_node(samples, subtrie)
    return _build_pb(samples, subtrie, params), params


def test_strict_prefix_junction_registered():
    """A=[1,2,3] is a strict prefix of B=[1,2,3,4]: the junction (last shared
    token, flat pos 2) must be registered with B's next token 4."""
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    _, params = _layout(tensors)
    assert params.boundary_registry == [(2, [(1, 4)])], params.boundary_registry


def test_fork_registry_unchanged():
    """Ordinary fork (no strict-prefix termination) must not gain entries:
    registry stays one boundary with all three leaves."""
    tensors = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 3, 6])]
    _, params = _layout(tensors)
    assert params.boundary_registry == [(2, [(0, 4), (1, 5), (2, 6)])], params.boundary_registry


def test_fork_plus_termination():
    """Fork where one branch also terminates mid-way: fork boundary carries all
    leaves; the terminating branch's junction carries only the continuing leaf."""
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5])]
    _, params = _layout(tensors)
    assert params.boundary_registry == [(2, [(1, 4), (2, 5)])], params.boundary_registry


def test_chain_junctions_registered():
    """A=[1,2] ⊂ B=[1,2,3] ⊂ C=[1,2,3,4]: junction 1 is read by BOTH B and C
    (B terminates at the internal node [3], not a childless leaf); junction 2
    is read only by C."""
    tensors = [torch.tensor([1, 2]), torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    _, params = _layout(tensors)
    assert params.boundary_registry == [(1, [(1, 3), (2, 3)]), (2, [(2, 4)])], params.boundary_registry


def test_strict_prefix_junction_restore():
    """End-to-end restore: only the continuing sample B has a boundary entry
    (A terminates — its label at the junction is the masked 0-pad, so A keeps
    the flat value). B's tensor must carry B's own log-prob at the junction."""
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    pb, params = _layout(tensors)
    (boundary_pos, [(sample_idx, _next_token)]) = params.boundary_registry[0]
    assert (boundary_pos, sample_idx) == (2, 1)

    b_val = torch.tensor(-7.0)
    pb._boundary_logps = {sample_idx: [(boundary_pos, b_val)]}

    flat = torch.arange(pb.tree_packed_input_ids.shape[0], dtype=torch.float32)
    restored = restore_flat_to_nested(flat, pb, apply_boundary_patch=True)
    lengths = restored.offsets().diff().tolist()
    assert lengths == [len(s) for s in tensors]
    vals = restored.values()
    # A (sample 0): no entry → keeps flat value at the junction.
    assert vals[2] == flat[boundary_pos], "terminating sample's junction row was modified"
    assert vals[0] == flat[0] and vals[1] == flat[1], "terminating sample's prefix rows corrupted"
    # B (sample 1): junction patched to its own log-prob.
    assert vals[3 + 2] == b_val, "continuing sample's junction not patched"
    assert vals[3 + 3] == flat[3], "continuing sample's extension row corrupted"
