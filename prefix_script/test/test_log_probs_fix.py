"""GPU test: verify MAGI log_probs match expected per-sample labels after fix."""

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


def make_batch(tokens_list):
    """Build a MAGI batch from per-sample token lists."""
    tokens = [torch.tensor(t) for t in tokens_list]
    shared, children = build_tree_dynamic(tokens)
    assert shared is not None
    params = build_layout_from_tree_node(tokens, shared, children)
    return PrefixTreeMagiBatch(
        flat_input_ids=params.flat_tokens,
        flat_position_ids=params.flat_position_ids,
        flat_loss_mask=None,
        magi_key=None,
        flex_key=None,
        leaf_to_sample=params.leaf_to_sample,
        leaf_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=params.num_samples,
    ), tokens


def test_log_probs_match():
    """Test that per-sample-rolled labels give correct log_probs."""
    # 2 samples sharing prefix [1,2,3], responses [10,11] and [20,21]
    tokens_list = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]
    pt, orig_tokens = make_batch(tokens_list)

    real = pt.real_tokens
    vocab_size = 100

    # Asymmetric logits: at prefix end (pos 2), favor token 10 over 20
    # This exposes the bug: old label=10 (wrong for s1), new label=20 (correct)
    flat_logits = torch.zeros(real, vocab_size)
    for i in range(real):
        flat_logits[i, pt.flat_input_ids[i]] = 1.0
    # At last prefix position (pos 2), give different scores
    flat_logits[2, 10] = 3.0  # model strongly predicts token 10
    flat_logits[2, 20] = 1.0  # model weakly predicts token 20

    # --- Method 1: NEW (restore -> per-sample roll -> flatten -> compute) ---
    nested_logits, nested_labels = _restore_and_roll_labels(flat_logits, pt.flat_input_ids[:real], pt)
    flat_new_logits = nested_logits.values().unsqueeze(1)
    flat_new_label = nested_labels.values().unsqueeze(1)
    new_log_probs_flat = F.log_softmax(flat_new_logits.squeeze(1).float(), dim=-1)[
        torch.arange(flat_new_label.shape[0]), flat_new_label.squeeze(1).long()
    ]
    # Restore to nested
    cu = nested_logits.offsets()
    new_nested = torch.nested.nested_tensor_from_jagged(new_log_probs_flat, cu)

    # --- Method 2: OLD (flat roll with boundary fix) ---
    old_label = torch.roll(pt.flat_input_ids[:real], shifts=-1, dims=0)
    for _, le in pt.leaf_ranges:
        if le <= real:
            old_label[le - 1] = pt.flat_input_ids[le - 1]
    old_label = old_label.unsqueeze(1)
    old_logits = flat_logits.unsqueeze(1)
    old_log_probs_flat = F.log_softmax(old_logits.squeeze(1).float(), dim=-1)[
        torch.arange(old_label.shape[0]), old_label.squeeze(1).long()
    ]
    old_nested = restore_flat_to_nested(old_log_probs_flat, pt)

    # --- Method 3: EXPECTED (per-sample manual roll) ---
    expected = []
    for tok in orig_tokens:
        rolled = torch.roll(tok, shifts=-1, dims=0)
        lp = F.log_softmax(flat_logits[:real].float(), dim=-1)  # simplified: use flat logits
        # For each position in this sample, the logit is at the corresponding flat position
        # This test uses a simplified model where logits are token-matched
        expected_lp = []
        # Map per-sample positions to flat positions using the restore function
        pass  # We validate by comparing old vs new

    # Compare old vs new per-sample
    prompt_len = 3
    resp_len = 2
    all_ok = True
    for s in range(len(orig_tokens)):
        new_sample = new_nested[s]
        old_sample = old_nested[s]
        # Response labels (positions [prompt_len-1 : prompt_len+resp_len-1])
        new_resp = new_sample[prompt_len - 1 : prompt_len + resp_len - 1]
        old_resp = old_sample[prompt_len - 1 : prompt_len + resp_len - 1]
        expected_resp_tokens = orig_tokens[s][prompt_len : prompt_len + resp_len]

        print(f"Sample {s}:")
        print(f"  NEW  resp log_probs: {new_resp.tolist()}")
        print(f"  OLD  resp log_probs: {old_resp.tolist()}")
        print(f"  Expected tokens:      {expected_resp_tokens.tolist()}")

        # Verify NEW matches expected labels
        for j, exp_tok in enumerate(expected_resp_tokens):
            # flat position for boundary: pos 2 (last prefix) → flat_logits[2]
            if j == 0:
                correct_logit = flat_logits[2]  # boundary prediction
            else:
                # Non-boundary: sample 0 uses flat[3], sample 1 uses flat[5]
                correct_logit = flat_logits[3 + s * 2 + j - 1]
            expected_lp = F.log_softmax(correct_logit.float().unsqueeze(0), dim=-1)[0, exp_tok]

            new_lp = new_resp[j].item()
            old_lp = old_resp[j].item()
            new_ok = abs(new_lp - expected_lp.item()) < 1e-5
            old_ok = abs(old_lp - expected_lp.item()) < 1e-5
            all_ok = all_ok and new_ok
            status_new = "✓" if new_ok else "✗"
            status_old = "✓" if old_ok else "✗"
            print(
                f"    pos {j}: token={exp_tok} expected_lp={expected_lp.item():.6f} new={new_lp:.6f} {status_new} old={old_lp:.6f} {status_old}"
            )

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL - new method has wrong log_probs'}")


if __name__ == "__main__":
    test_log_probs_match()
