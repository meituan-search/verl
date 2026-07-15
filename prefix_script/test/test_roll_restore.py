# Copyright 2025-2026 Meituan Ltd. and/or its affiliates
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

"""Unit test: verify MAGI flat roll + restore produces correct per-sample labels."""

import sys

sys.path.insert(0, ".")
import torch

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import PrefixTreeMagiBatch
from verl.utils.prefix_tree.utils import build_layout_from_tree_node

# ============================================================
# SINGLE-LEVEL TREE: 2 samples sharing prompt, different responses
# Sample 0: [1,2,3,10,11]  (prompt=3, response=2)
# Sample 1: [1,2,3,20,21]  (prompt=3, response=2)
# ============================================================
tokens = [torch.tensor([1, 2, 3, 10, 11]), torch.tensor([1, 2, 3, 20, 21])]
shared, children = build_tree_dynamic(tokens)
assert shared is not None
params = build_layout_from_tree_node(tokens, shared, children)

pt = PrefixTreeMagiBatch(
    flat_input_ids=params.flat_tokens,
    flat_position_ids=params.flat_position_ids,
    flat_loss_mask=None,
    magi_key=None,
    flex_key=None,
    leaf_to_sample=params.leaf_to_sample,
    leaf_ranges=params.leaf_ranges,
    prefix_range=params.prefix_range,
    original_batch_size=params.num_samples,
)

print("=== SINGLE-LEVEL TREE ===")
print("Flat tokens:", pt.flat_input_ids.tolist())
print("prefix_range:", pt.prefix_range)
print("leaf_ranges:", pt.leaf_ranges)

# Simulate MAGI roll + boundary fix (magi.py:579-586)
real = pt.real_tokens
flat_label = torch.roll(pt.flat_input_ids[:real], shifts=-1, dims=0)
print("After roll(-1):", flat_label.tolist())
for _, leaf_end in pt.leaf_ranges:
    if leaf_end <= real:
        flat_label[leaf_end - 1] = pt.flat_input_ids[leaf_end - 1]
print("After boundary fix:", flat_label.tolist())

# Restore per-sample and check response labels
prompt_len, resp_len = 3, 2
ps, pe = pt.prefix_range
p_lab = flat_label[ps:pe]

all_ok = True
for li in range(len(pt.leaf_ranges)):
    sid = pt.leaf_to_sample[li]
    ls, le = pt.leaf_ranges[li]
    leaf_lab = flat_label[ls:le]
    restored = torch.cat([p_lab, leaf_lab])
    magi_resp = restored[prompt_len - 1 : prompt_len + resp_len - 1]
    expected = tokens[sid][prompt_len : prompt_len + resp_len]
    ok = torch.equal(magi_resp, expected)
    all_ok = all_ok and ok
    print(f"Sample {sid}: MAGI={magi_resp.tolist()} expected={expected.tolist()} OK={ok}")

print("SINGLE-LEVEL:", "PASS" if all_ok else "FAIL")

# ============================================================
# MULTI-LEVEL TREE: 2 samples, 2 turns
# Sample 0: [1,2,3, 10,11, 20,21]  (prompt=3, turn1=2, turn2=2)
# Sample 1: [1,2,3, 10,11, 30,31]
# ============================================================
tokens2 = [
    torch.tensor([1, 2, 3, 10, 11, 20, 21]),
    torch.tensor([1, 2, 3, 10, 11, 30, 31]),
]
shared2, children2 = build_tree_dynamic(tokens2)
assert shared2 is not None
params2 = build_layout_from_tree_node(tokens2, shared2, children2)

pt2 = PrefixTreeMagiBatch(
    flat_input_ids=params2.flat_tokens,
    flat_position_ids=params2.flat_position_ids,
    flat_loss_mask=None,
    magi_key=None,
    flex_key=None,
    leaf_to_sample=params2.leaf_to_sample,
    leaf_ranges=params2.leaf_ranges,
    prefix_range=params2.prefix_range,
    original_batch_size=params2.num_samples,
)

print()
print("=== MULTI-LEVEL TREE ===")
print("Flat tokens:", pt2.flat_input_ids.tolist())
print("prefix_range:", pt2.prefix_range)
print("leaf_ranges:", pt2.leaf_ranges)
print("leaf_to_sample:", pt2.leaf_to_sample)
if pt2.leaf_ancestor_ranges is not None:
    print("ancestor_ranges:", pt2.leaf_ancestor_ranges)

real2 = pt2.real_tokens
flat2 = torch.roll(pt2.flat_input_ids[:real2], shifts=-1, dims=0)
for _, le in pt2.leaf_ranges:
    if le <= real2:
        flat2[le - 1] = pt2.flat_input_ids[le - 1]
print("Rolled+fixed:", flat2.tolist())

ps2, pe2 = pt2.prefix_range
p_lab2 = flat2[ps2:pe2]

all_ok2 = True
for li in range(len(pt2.leaf_ranges)):
    sid = pt2.leaf_to_sample[li]
    ls, le = pt2.leaf_ranges[li]
    leaf_lab = flat2[ls:le]
    if pt2.leaf_ancestor_ranges is not None:
        parts = [flat2[s:e] for s, e in pt2.leaf_ancestor_ranges[li]]
        parts.append(leaf_lab)
        restored = torch.cat(parts)
    else:
        restored = torch.cat([p_lab2, leaf_lab])

    orig = tokens2[sid]
    resp_len2 = 4  # turn1=2 + turn2=2
    plen2 = len(orig) - resp_len2
    magi_resp = restored[plen2 - 1 : plen2 + resp_len2 - 1]
    expected = orig[plen2 : plen2 + resp_len2]
    ok = torch.equal(magi_resp, expected)
    all_ok2 = all_ok2 and ok
    print(f"Sample {sid}: MAGI={magi_resp.tolist()} expected={expected.tolist()} OK={ok}")

print("MULTI-LEVEL:", "PASS" if all_ok2 else "FAIL")
print()
print("OVERALL:", "PASS" if (all_ok and all_ok2) else "FAIL")
