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

"""Quantify log_prob diff: old flat roll vs new per-sample roll vs reference.

Simulates `calculate_debug_metrics` comparison:
  rollout_log_probs (reference) vs old_log_probs (MAGI).

With mock logits that have realistic vocabulary-favoring, shows the e-1 diff
caused by the flat roll bug, and how the fix corrects it.
"""

import sys

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import (
    PrefixTreeMagiBatch,
    _restore_and_roll_labels,
    restore_flat_to_nested,
)
from verl.utils.prefix_tree.utils import build_layout_from_tree_node

# Simulate a more realistic batch: 4 samples sharing prompt, rollout_n=4
# Each sample has prompt [1,2,3,4,5] and 8 unique response tokens
prompt_len = 5
resp_len = 8
vocab_size = 1000
n_samples = 4

# Create samples with shared prefix and distinct responses
all_tokens = []
for s in range(n_samples):
    prompt = [1, 2, 3, 4, 5]
    response = list(range(100 + s * 100, 100 + s * 100 + resp_len))
    all_tokens.append(prompt + response)

# Build MAGI batch
torch_tokens = [torch.tensor(t) for t in all_tokens]
shared, children = build_tree_dynamic(torch_tokens)
assert shared is not None
params = build_layout_from_tree_node(torch_tokens, shared, children)
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

real = pt.real_tokens

# Create realistic logits: model strongly favors correct next-token (logit=5.0),
# random other tokens get small values (logit ~ N(0,0.5))
torch.manual_seed(42)
flat_logits = torch.randn(real, vocab_size) * 0.5
for i in range(real):
    if i + 1 < pt.flat_input_ids.shape[0]:
        flat_logits[i, pt.flat_input_ids[i + 1]] = 5.0

# --- REFERENCE: per-sample roll (what FA3/vLLM would produce) ---
ref_log_probs = []
for s in range(n_samples):
    tok = torch.tensor(all_tokens[s])
    rolled = torch.roll(tok[: prompt_len + resp_len], shifts=-1, dims=0)  # next-token labels
    # Each position's logit comes from the corresponding flat position
    # We need to map per-sample positions to flat positions
    # sample s positions 0..prompt_len-1 map to flat positions 0..prompt_len-1
    # sample s positions prompt_len..prompt_len+resp_len-1 map to leaf positions
    lp = []
    for j in range(prompt_len + resp_len - 1):
        if j < prompt_len - 1:
            flat_pos = j
        elif j == prompt_len - 1:
            flat_pos = prompt_len - 1  # all samples share this boundary logit
        else:
            leaf_flat_start = pt.leaf_ranges[s][0]
            leaf_offset = j - prompt_len
            flat_pos = leaf_flat_start + leaf_offset
        logit = flat_logits[flat_pos]
        label = rolled[j]
        lp.append(F.log_softmax(logit.float(), dim=-1)[label].item())
    # Pad to response_len (for compatibility with debug metrics format)
    ref_log_probs.append(torch.tensor(lp[-resp_len:]))

# --- OLD: flat roll (buggy) ---
old_label = torch.roll(pt.flat_input_ids[:real], shifts=-1, dims=0)
for _, le in pt.leaf_ranges:
    if le <= real:
        old_label[le - 1] = pt.flat_input_ids[le - 1]
old_flat_lp = F.log_softmax(flat_logits.float(), dim=-1)[torch.arange(real), old_label.long()]
old_nested = restore_flat_to_nested(old_flat_lp, pt)
old_log_probs = []
for s in range(n_samples):
    seq = old_nested[s]
    old_log_probs.append(seq[-resp_len:])

# --- NEW: per-sample roll (fixed) ---
_, new_labels = _restore_and_roll_labels(flat_logits, pt.flat_input_ids[:real], pt)
new_flat_logits = _restore_and_roll_labels(flat_logits, pt.flat_input_ids[:real], pt)[0].values()
new_flat_label = new_labels.values()
new_flat_lp = F.log_softmax(new_flat_logits.float(), dim=-1)[
    torch.arange(new_flat_label.shape[0]), new_flat_label.long()
]
cu = pt.flat_input_ids[:real].offsets()  # wrong, need nested offsets
# Better: compute directly
new_log_probs = []
nested_logits, nested_labels = _restore_and_roll_labels(flat_logits, pt.flat_input_ids[:real], pt)
for s in range(n_samples):
    logit = nested_logits[s]
    label = nested_labels[s]
    lp_s = F.log_softmax(logit.float(), dim=-1)[torch.arange(label.shape[0]), label.long()]
    new_log_probs.append(lp_s[-resp_len:])

# --- Compare ---
print("=" * 70)
print("Log_prob comparison: OLD (flat roll) vs NEW (per-sample) vs REF (FA3)")
print("=" * 70)

total_ref_old_diff = 0.0
total_ref_new_diff = 0.0
total_tokens = 0

for s in range(n_samples):
    ref_lp = torch.tensor(ref_log_probs[s])
    old_lp = old_log_probs[s]
    new_lp = new_log_probs[s]
    old_diff = (ref_lp - old_lp).abs()
    new_diff = (ref_lp - new_lp).abs()
    total_ref_old_diff += old_diff.sum().item()
    total_ref_new_diff += new_diff.sum().item()
    total_tokens += resp_len
    print(f"Sample {s} (first resp token={all_tokens[s][prompt_len]}):")
    print(f"  REF  log_probs: {[f'{x:.4f}' for x in ref_lp.tolist()]}")
    print(f"  OLD  log_probs: {[f'{x:.4f}' for x in old_lp.tolist()]}")
    print(f"  NEW  log_probs: {[f'{x:.4f}' for x in new_lp.tolist()]}")
    print(f"  |REF-OLD| mean: {old_diff.mean().item():.4f}  max: {old_diff.max().item():.4f}")
    print(f"  |REF-NEW| mean: {new_diff.mean().item():.4f}  max: {new_diff.max().item():.4f}")

print("\n--- AGGREGATE (simulating calculate_debug_metrics) ---")
print(f"Total tokens: {total_tokens}")
print(f"|REF-OLD| mean: {total_ref_old_diff / total_tokens:.6f}  ({total_ref_old_diff / total_tokens:.1e})")
print(f"|REF-NEW| mean: {total_ref_new_diff / total_tokens:.6f}  ({total_ref_new_diff / total_tokens:.1e})")
print(f"Improvement: {total_ref_old_diff / total_ref_new_diff:.1f}x")
print(f"\nOLD diff is at {total_ref_old_diff / total_tokens:.1e} level (bug)")
print(f"NEW diff is at {total_ref_new_diff / total_tokens:.1e} level (fixed)")
