"""Test: does restore_flat_to_nested produce correct per-sample lengths?"""

import sys

sys.path.insert(0, ".")
import torch

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.utils import build_layout_from_tree_node

# Single-level: 2 samples, shared prompt
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

# Restore a dummy tensor (use token IDs as values for easy inspection)
flat = pt.flat_input_ids.float()
restored = restore_flat_to_nested(flat[: pt.real_tokens], pt)

print("SINGLE-LEVEL LENGTH CHECK:")
print(f"  flat tokens: {pt.flat_input_ids.tolist()}")
for i, orig in enumerate(tokens):
    r_len = restored[i].shape[0]
    o_len = orig.shape[0]
    ok = r_len == o_len
    print(f"  sample {i}: restored_len={r_len} original_len={o_len} {'OK' if ok else 'MISMATCH'}")

# Multi-level: 2 samples, 2 turns
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

flat2 = pt2.flat_input_ids.float()
restored2 = restore_flat_to_nested(flat2[: pt2.real_tokens], pt2)

print("\nMULTI-LEVEL LENGTH CHECK:")
print(f"  flat tokens: {pt2.flat_input_ids.tolist()}")
for i, orig in enumerate(tokens2):
    r_len = restored2[i].shape[0]
    o_len = orig.shape[0]
    ok = r_len == o_len
    # Also check: does restored content differ from original?
    r_vals = restored2[i].long().tolist()
    o_vals = orig.tolist()
    content_ok = r_vals == o_vals
    print(f"  sample {i}: restored_len={r_len} original_len={o_len} {'OK' if ok else 'MISMATCH'}")
    if not content_ok:
        for j in range(min(len(r_vals), len(o_vals))):
            if r_vals[j] != o_vals[j]:
                print(f"    pos {j}: restored={r_vals[j]} original={o_vals[j]}")

print(
    "\nOVERALL:",
    "PASS"
    if all(restored[i].shape[0] == tokens[i].shape[0] for i in range(len(tokens)))
    and all(restored2[i].shape[0] == tokens2[i].shape[0] for i in range(len(tokens2)))
    else "FAIL - length mismatch detected",
)
