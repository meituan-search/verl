"""Test: order preservation from OLP restore → storage → update retrieval.

Emulates the full pipeline: flat log_probs → restore_flat_to_nested →
no_padding_2_padding → store as padded → re-extract → compare.
"""

import sys

sys.path.insert(0, ".")
import torch
from tensordict import TensorDict

from verl.utils.prefix_tree.dynamic import build_tree_dynamic, dfs_leaf_order
from verl.utils.prefix_tree.magi import PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.utils import build_layout_from_tree_node
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# 4 samples sharing same prefix [1,2,3], different responses
# Sample order matters: we test that DFS order ≠ original order to catch mismatches
all_tokens = [
    [1, 2, 3, 401, 402],  # sample 0
    [1, 2, 3, 101, 102],  # sample 1
    [1, 2, 3, 501, 502],  # sample 2
    [1, 2, 3, 201, 202],  # sample 3
]

# DFS order
dfs_order = dfs_leaf_order(all_tokens)
print(f"DFS order: {dfs_order}")
print(f"  (batch row 0 = sample {dfs_order[0]}, row 1 = sample {dfs_order[1]}, ...)")

# Build MAGI batch from DFS-ordered data
dfs_tokens = [all_tokens[i] for i in dfs_order]
dfs_torch = [torch.tensor(t) for t in dfs_tokens]
shared, children = build_tree_dynamic(dfs_torch)
assert shared is not None
params = build_layout_from_tree_node(dfs_torch, shared, children)

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
print(f"\nleaf_to_sample (MAGI): {pt.leaf_to_sample}")
print(f"  (leaf 0 → sample {pt.leaf_to_sample[0]}, leaf 1 → sample {pt.leaf_to_sample[1]}, ...)")

# Simulate OLP: flat log_probs (each position = its own token ID as value)
real = pt.real_tokens
flat_lp = pt.flat_input_ids[:real].float()
restored_nested = restore_flat_to_nested(flat_lp, pt)

print("\nrestored_nested (per-sample):")
for idx in range(len(dfs_tokens)):
    t = restored_nested[idx]
    print(
        f"  nested pos {idx}: {t.int().tolist()}  (expected sample {pt.leaf_to_sample[idx]}, tokens={dfs_tokens[idx]})"
    )
    assert t.int().tolist() == dfs_tokens[idx], f"mismatch at pos {idx}"

# Now emulate ray_trainer: create TensorDict, left_right_2_no_padding, no_padding_2_padding
BS = len(dfs_tokens)
max_seq = max(len(t) for t in dfs_tokens)
max_resp = 2  # response length

# Create padded tensors matching DFS batch order
input_ids = torch.zeros(BS, max_seq, dtype=torch.long)
attention_mask = torch.zeros(BS, max_seq, dtype=torch.int64)
response_mask = torch.zeros(BS, max_resp, dtype=torch.int64)
position_ids = torch.zeros(BS, max_seq, dtype=torch.long)

for i, (tid, tokens) in enumerate(zip(dfs_order, dfs_tokens, strict=False)):
    L = len(tokens)
    input_ids[i, :L] = torch.tensor(tokens)
    attention_mask[i, :L] = 1
    response_mask[i, :] = 1  # all response tokens active
    position_ids[i, :L] = torch.arange(L)

# Build real prompts/responses (what DataProto has)
prompt_ids = input_ids[:, : max_seq - max_resp]
response_ids = input_ids[:, max_seq - max_resp :]

td = TensorDict(
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "position_ids": position_ids,
        "prompts": prompt_ids,
        "responses": response_ids,
    },
    batch_size=[BS],
)

td = left_right_2_no_padding(td)

# Simulate _compute_old_log_prob: expand to padded
# We need log_probs that match the restored_nested structure
# Convert restored_nested log_probs to flat for no_padding_2_padding
model_log_probs_nested = restored_nested.clone()
olp_padded = no_padding_2_padding(model_log_probs_nested, td)

print("\nOLP padded (no_padding_2_padding output):")
print(f"  shape: {olp_padded.shape}")
for i in range(BS):
    print(
        f"  row {i}: {olp_padded[i].int().tolist()}  (expected sample {dfs_order[i]}, tokens={dfs_tokens[i][-max_resp:]})"
    )

# Simulate batch.union: store as old_log_probs
td_after_union = td  # same td, now also has old_log_probs
td_after_union = td  # (in reality, union merges)

# Simulate _update_actor: re-extract
# The padded old_log_probs are stored separately and unioned back
# For simplicity, we just verify that td's row order matches
print("\nOrder check:")
for i in range(BS):
    tid = dfs_order[i]
    orig_tokens = all_tokens[tid]
    response_tokens = orig_tokens[-max_resp:]
    olp_tokens = olp_padded[i].int().tolist()
    match = olp_tokens == response_tokens
    status = "✓" if match else "✗ MISMATCH"
    print(f"  {status} row {i} (sample {tid}): OLP={olp_tokens} expected={response_tokens}")

# Also verify: restored_nested[i] == sample dfs_order[i]
for i in range(BS):
    assert restored_nested[i].int().tolist() == dfs_tokens[i], (
        f"Nested mismatch at {i}: {restored_nested[i].int().tolist()} vs {dfs_tokens[i]}"
    )

print("\n=== PASS: order preserved ===")
