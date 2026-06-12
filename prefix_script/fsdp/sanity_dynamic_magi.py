# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""dynamic-trie + Magi single-GPU sanity: forward / backward equivalence + mode determinism.

Three checks on a GRPO-style batch (1 prompt × N rollouts) using Qwen2.5-0.5B.
Works for both plain HF (default) and FSDP2-wrapped models (``USE_FSDP=1``):

  Check A — forward equivalence:
      dynamic-trie + Magi packed forward vs dense (sdpa) per-sample forward must agree
      within bf16 noise floor.

  Check B — backward equivalence:
      dynamic-trie + Magi loss.backward() vs dense loss.backward() must produce the
      same grad_norm (relative diff within tolerance).

  Check C — mode determinism (EVAL/TRAIN/EVAL):
      dynamic-trie + Magi forward in {EVAL no_grad, TRAIN grad, EVAL no_grad again}
      must produce identical outputs. Catches state leakage between
      mode switches — originally the symptom that drove the LRU-order
      investigation; now expected to be deterministic.

Env tunables:
  USE_FSDP=1       wrap the model in FSDP2 (matches production verl) —
                   if you only have time for one sanity, use this one.
  MODEL_PATH       HF model dir (default ~/models/Qwen/Qwen2.5-0.5B-Instruct)
  N, P, R          rollouts / prompt-len / response-len (defaults 4/64/32)

Usage:
    torchrun --standalone --nproc_per_node=1 prefix_script/fsdp/sanity_dynamic_magi.py
    USE_FSDP=1 torchrun --standalone --nproc_per_node=1 prefix_script/fsdp/sanity_dynamic_magi.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

N = int(os.environ.get("N", "4"))
P = int(os.environ.get("P", "64"))
R = int(os.environ.get("R", "32"))
USE_FSDP = os.environ.get("USE_FSDP", "0") == "1"


def _init_dist():
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            raise RuntimeError("Run via torchrun")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")


def _build_fsdp_qwen(model_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers import Qwen2ForCausalLM

    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

    model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=torch.float32).cuda()
    world = dist.get_world_size()
    mesh = init_device_mesh("cuda", mesh_shape=(world,), mesh_dim_names=("fsdp",))
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    full_state = model.state_dict()
    apply_fsdp2(
        model,
        {"mesh": mesh, "mp_policy": mp, "offload_policy": None, "reshard_after_forward": True},
        config={},
    )
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    return model


def _load_model(model_path, cp_group):
    from transformers import Qwen2ForCausalLM

    from verl.models.transformers.monkey_patch import apply_magi_prefix_tree_backend_fsdp

    apply_magi_prefix_tree_backend_fsdp()

    if USE_FSDP:
        model = _build_fsdp_qwen(model_path)
    else:
        model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
    model.train()
    model.config._attn_implementation = "Magi_Attention"
    model.config.attention_dropout = 0.0

    n_attached = 0
    for _, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if cls.endswith(("attention", "self_attn", "selfattention")):
            mod.cp_group = cp_group
            n_attached += 1
    print(f"[SANITY] Attached cp_group to {n_attached} attention modules")
    return model


def _build_batch(model):
    torch.manual_seed(42)
    vocab = model.config.vocab_size
    prompt = torch.randint(0, vocab, (P,), device="cuda")
    samples = []
    for i in range(N):
        torch.manual_seed(100 + i)
        response = torch.randint(0, vocab, (R,), device="cuda")
        samples.append(torch.cat([prompt, response]))
    nested = torch.nested.nested_tensor(samples, layout=torch.jagged)
    return samples, nested


def _dyn_forward(model, nested, cp_group):
    """dynamic-trie + Magi packed forward; returns per-sample logits list."""
    from verl.models.transformers.monkey_patch import set_magi_attention_key_fsdp
    from verl.utils.prefix_tree_magi import build_prefix_tree_micro_batch, restore_flat_to_nested

    pt_batch = build_prefix_tree_micro_batch(model, nested, attention_type="magi", cp_size=1, dynamic_trie=True)
    assert pt_batch is not None
    set_magi_attention_key_fsdp(model, pt_batch.magi_key)
    flat_in = pt_batch.local_flat_input_ids.unsqueeze(0)
    flat_pos = pt_batch.local_flat_position_ids.unsqueeze(0)
    out = model(input_ids=flat_in, attention_mask=None, position_ids=flat_pos, use_cache=False)
    flat_logits = out.logits[0][: pt_batch.real_tokens]
    nested_logits = restore_flat_to_nested(flat_logits, pt_batch)
    return [nested_logits[i] for i in range(pt_batch.original_batch_size)]


def _dense_forward(model, samples):
    """Per-sample dense (sdpa) forward."""
    model.config._attn_implementation = "sdpa"
    try:
        return [model(input_ids=s.unsqueeze(0), attention_mask=None, use_cache=False).logits[0] for s in samples]
    finally:
        model.config._attn_implementation = "Magi_Attention"


def _loss(per_sample, samples):
    """Mean negative log-prob over each sample's response tokens."""
    losses = []
    for logits, s in zip(per_sample, samples, strict=False):
        f = logits.float()
        labels = s[1:]
        logp = torch.log_softmax(f[:-1], dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        losses.append(-logp.mean())
    return torch.stack(losses).mean()


def _grad_norm(model):
    """Global L2 grad norm. Handles FSDP2 DTensor grads (convert to local, then
    all-reduce sum-of-squares across ranks)."""
    total_sq = torch.zeros((), device="cuda")  # scalar (shape ()), not (1,) — avoids DTensor wrap
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            # FSDP2 wraps p.grad as DTensor; unwrap to local shard before computation.
            if hasattr(g, "to_local"):
                g = g.to_local()
            total_sq = total_sq + g.float().pow(2).sum()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    return total_sq.sqrt().item()


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b, strict=False))


def main() -> int:
    _init_dist()
    cp_group = dist.group.WORLD

    model_path = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct"))
    print(f"[SANITY] USE_FSDP={USE_FSDP}  N={N} P={P} R={R}")
    print(f"[SANITY] Loading {model_path}")
    model = _load_model(model_path, cp_group)
    samples, nested = _build_batch(model)
    total_tokens = sum(s.shape[0] for s in samples)
    print(f"[SANITY] total_tokens={total_tokens}, prefix_shared={P * (N - 1)}")

    BOLD, GREEN, RED, RESET = "\033[1m", "\033[0;32m", "\033[0;31m", "\033[0m"
    fail = []

    # ──────────────────────────────────────────────────────────────
    # Check A: forward equivalence (dynamic-trie + Magi vs dense)
    # ──────────────────────────────────────────────────────────────
    print()
    print("[SANITY] Check A: forward equivalence (dynamic-trie + Magi vs dense sdpa)")
    with torch.no_grad():
        dyn_logits = _dyn_forward(model, nested, cp_group)
        dense_logits = _dense_forward(model, samples)
    fwd_diff = _max_diff(dyn_logits, dense_logits)
    fwd_threshold = 1.0  # bf16 noise floor at Qwen2.5-0.5B logit scale
    print(f"[SANITY]   max_diff = {fwd_diff:.4f}  (threshold {fwd_threshold})")
    if fwd_diff < fwd_threshold:
        print(f"[SANITY]   {GREEN}check A: PASS{RESET}")
    else:
        print(f"[SANITY]   {RED}check A: FAIL{RESET}")
        fail.append(f"forward max_diff={fwd_diff:.4f}")

    # ──────────────────────────────────────────────────────────────
    # Check B: backward equivalence (grad_norm rel diff)
    # ──────────────────────────────────────────────────────────────
    print()
    print("[SANITY] Check B: backward grad_norm equivalence")
    model.zero_grad(set_to_none=True)
    dyn_logits = _dyn_forward(model, nested, cp_group)
    _loss(dyn_logits, samples).backward()
    norm_dyn = _grad_norm(model)

    model.zero_grad(set_to_none=True)
    dense_logits = _dense_forward(model, samples)
    _loss(dense_logits, samples).backward()
    norm_dense = _grad_norm(model)

    rel_diff = abs(norm_dyn - norm_dense) / max(norm_dense, 1e-9)
    norm_threshold = 0.1  # 10% rel — bf16 + per-sample-vs-packed path diff
    print(
        f"[SANITY]   grad_norm dyn={norm_dyn:.4f}  dense={norm_dense:.4f}  "
        f"rel={rel_diff:.4f}  (threshold {norm_threshold})"
    )
    if rel_diff < norm_threshold and torch.isfinite(torch.tensor(norm_dyn)):
        print(f"[SANITY]   {GREEN}check B: PASS{RESET}")
    else:
        print(f"[SANITY]   {RED}check B: FAIL{RESET}")
        fail.append(f"grad_norm rel={rel_diff:.4f}")

    # ──────────────────────────────────────────────────────────────
    # Check C: EVAL → TRAIN → EVAL determinism
    # ──────────────────────────────────────────────────────────────
    print()
    print("[SANITY] Check C: EVAL/TRAIN/EVAL mode determinism")
    model.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        out_eval_1 = [t.detach().clone() for t in _dyn_forward(model, nested, cp_group)]

    model.train()
    out_train = [t.detach().clone() for t in _dyn_forward(model, nested, cp_group)]

    model.eval()
    with torch.no_grad():
        out_eval_2 = [t.detach().clone() for t in _dyn_forward(model, nested, cp_group)]

    eval_train_diff = _max_diff(out_eval_1, out_train)
    eval_eval_diff = _max_diff(out_eval_1, out_eval_2)
    mode_threshold = 1.0  # bf16 noise
    print(f"[SANITY]   diff(EVAL_1, TRAIN)  = {eval_train_diff:.4f}")
    print(f"[SANITY]   diff(EVAL_1, EVAL_2) = {eval_eval_diff:.4f}")
    if eval_train_diff < mode_threshold and eval_eval_diff < mode_threshold:
        print(f"[SANITY]   {GREEN}check C: PASS{RESET}")
    else:
        print(f"[SANITY]   {RED}check C: FAIL{RESET}")
        fail.append(f"mode diff EVAL/TRAIN={eval_train_diff:.4f}  EVAL/EVAL={eval_eval_diff:.4f}")

    # ──────────────────────────────────────────────────────────────
    # Verdict
    # ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if not fail:
        print(f"  {GREEN}{BOLD}[SANITY PASS] all 3 checks succeeded ({'FSDP2' if USE_FSDP else 'plain HF'}).{RESET}")
    else:
        print(f"  {RED}{BOLD}[SANITY FAIL]{RESET}")
        for f in fail:
            print(f"  {RED}  - {f}{RESET}")
    print("=" * 70)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
