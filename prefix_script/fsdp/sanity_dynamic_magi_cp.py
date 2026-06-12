# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sanity test for dynamic-trie + Magi forward under context-parallel (cp_size > 1).

Verifies the module-attribute key-threading fix (commit b6e76b41) on the
CP dispatch path: cp_size > 1 invokes ``dispatch(...)`` / ``undispatch(...)``
between build and forward, which also touches the Magi LRU. The fix should
work regardless of whether the LRU is touched or not — the attention func
reads the key from ``module._verl_magi_attention_key``, not from the LRU.

What we test
------------
1. cp_size=2 dynamic-trie + Magi forward on Qwen2.5-0.5B.
2. Multi-step LRU pollution to stress the wrong-key risk: build several
   shapes in a row, then re-forward an earlier shape. Under the old LRU
   path this would silently use the wrong mgr; under module-attribute
   threading the output stays correct.
3. Compare the cp_size=2 packed forward against a single-rank dense
   (sdpa) baseline. bf16 noise floor ~ a few-tenths in logits.

Usage
-----
    torchrun --standalone --nproc_per_node=2 \\
        prefix_script/fsdp/sanity_dynamic_magi_cp.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def _maybe_init_dist():
    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Run via torchrun so RANK/WORLD_SIZE env vars are set")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")


def _make_nested(samples, device):
    return torch.nested.nested_tensor(
        [torch.tensor(s, dtype=torch.long, device=device) for s in samples],
        layout=torch.jagged,
    )


def _dyn_magi_cp_forward(model, nested, cp_group):
    """One CP dynamic-trie + Magi forward; returns logits as a flat (total_real_tokens, V) tensor.

    Magi's ``dispatch`` / ``undispatch`` operate on a flat (seq,) tensor along
    dim=0. Mirror the production pattern: dispatch → unsqueeze for model, then
    squeeze → undispatch on the way out.
    """
    from verl.models.transformers.monkey_patch import set_magi_attention_key_fsdp
    from verl.utils.prefix_tree_magi import build_prefix_tree_micro_batch, restore_flat_to_nested

    pt_batch = build_prefix_tree_micro_batch(
        model,
        nested,
        attention_type="magi",
        cp_size=cp_group.size(),
        dynamic_trie=True,
    )
    if pt_batch is None:
        return None

    if cp_group.size() > 1:
        from magi_attention.api import dispatch

        local_in = dispatch(pt_batch.local_flat_input_ids, pt_batch.magi_key)
        local_pos = dispatch(pt_batch.local_flat_position_ids, pt_batch.magi_key)
    else:
        local_in = pt_batch.local_flat_input_ids
        local_pos = pt_batch.local_flat_position_ids

    flat_in = local_in.unsqueeze(0)
    flat_pos = local_pos.unsqueeze(0)

    set_magi_attention_key_fsdp(model, pt_batch.magi_key)
    with torch.no_grad():
        output = model(
            input_ids=flat_in,
            attention_mask=None,
            position_ids=flat_pos,
            use_cache=False,
        )

    logits = output.logits.squeeze(0)
    if cp_group.size() > 1:
        from magi_attention.api import undispatch

        logits = undispatch(logits, pt_batch.magi_key)

    flat_logits = logits[: pt_batch.real_tokens]
    nested_logits = restore_flat_to_nested(flat_logits, pt_batch)
    return [nested_logits[i].detach().clone() for i in range(pt_batch.original_batch_size)]


def _dense_forward(model, samples):
    """Per-sample dense (sdpa) forward; returns list of (L, V) logits."""
    model.config._attn_implementation = "sdpa"
    try:
        with torch.no_grad():
            out = []
            for s in samples:
                inp = torch.tensor([s], dtype=torch.long, device="cuda")
                logits = model(input_ids=inp, attention_mask=None, use_cache=False).logits[0]
                out.append(logits)
        return out
    finally:
        model.config._attn_implementation = "Magi_Attention"


def main() -> int:
    _maybe_init_dist()

    from transformers import Qwen2ForCausalLM

    from verl.models.transformers.monkey_patch import apply_magi_prefix_tree_backend_fsdp

    apply_magi_prefix_tree_backend_fsdp()

    model_path = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct"))
    if dist.get_rank() == 0:
        print(f"[CP-SANITY] Loading {model_path}")
    model = Qwen2ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()

    model.config._attn_implementation = "Magi_Attention"
    cp_group = dist.group.WORLD
    cp_size = cp_group.size()
    if cp_size < 2:
        if dist.get_rank() == 0:
            print(f"[CP-SANITY] SKIP: need WORLD_SIZE >= 2 to exercise CP, got {cp_size}")
        dist.destroy_process_group()
        return 0

    # Attach cp_group to each attention module — same as
    # FSDPEngine._attach_magi_cp_group_to_attention_modules.
    n_attached = 0
    for _, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if cls.endswith(("attention", "self_attn", "selfattention")):
            mod.cp_group = cp_group
            n_attached += 1
    if dist.get_rank() == 0:
        print(f"[CP-SANITY] cp_size={cp_size}, attached cp_group to {n_attached} attention modules")

    torch.manual_seed(42)
    P, R, N = 64, 32, 4
    prompt = torch.randint(0, model.config.vocab_size, (P,), device="cuda")
    samples = []
    for i in range(N):
        torch.manual_seed(100 + i)
        response = torch.randint(0, model.config.vocab_size, (R,), device="cuda")
        samples.append(torch.cat([prompt, response]).tolist())

    # ── Step 1: pollute the Magi LRU with a series of different-shape builds.
    # If the old LRU-side-channel path were still in use, the next REF
    # forward at the original shape would silently use whatever key was
    # last inserted, not the one matching its input. With module-attribute
    # threading, the right key is always set before forward.
    if dist.get_rank() == 0:
        print("[CP-SANITY] Polluting LRU with distractor builds ...")
    nested_main = _make_nested(samples, device="cuda")
    out_dyn_main = _dyn_magi_cp_forward(model, nested_main, cp_group)

    distractor_lengths = [80, 88, 96, 104, 112]
    for dl in distractor_lengths:
        torch.manual_seed(1000 + dl)
        d_samples = [torch.randint(0, model.config.vocab_size, (dl,), device="cuda").tolist() for _ in range(N)]
        _dyn_magi_cp_forward(model, _make_nested(d_samples, "cuda"), cp_group)

    if dist.get_rank() == 0:
        print("[CP-SANITY] Re-running REF after pollution ...")
    out_dyn_main_again = _dyn_magi_cp_forward(model, nested_main, cp_group)

    # ── Step 2: dense baseline (rank 0 only — sdpa doesn't need CP).
    if dist.get_rank() == 0:
        print("[CP-SANITY] Dense (sdpa) baseline ...")
        out_dense = _dense_forward(model, samples)
    else:
        out_dense = None

    # ── Compare on rank 0.
    if dist.get_rank() == 0:
        # 1) cp first vs cp re-run after pollution. Bf16 FFA kernel is NOT
        # deterministic across launches (reduce-order non-determinism),
        # so we don't require bit-exact — just bf16-noise level.
        # If module-attribute threading were broken (i.e. we were silently
        # reusing the wrong cached mgr like repro 7 demonstrates), the
        # drift would be huge (134%+ in repro 7's FULL-vs-CAUSAL setup).
        within_max = 0.0
        for a, b in zip(out_dyn_main, out_dyn_main_again, strict=False):
            within_max = max(within_max, (a - b).abs().max().item())
        print(f"[CP-SANITY] cp dynamic-trie first vs re-run after pollution: max_diff={within_max:.6f}")

        # 2) cp dynamic-trie vs dense (sdpa) reference.
        cp_vs_dense_max = 0.0
        for a, d in zip(out_dyn_main, out_dense, strict=False):
            cp_vs_dense_max = max(cp_vs_dense_max, (a - d).abs().max().item())
        print(f"[CP-SANITY] cp dynamic-trie vs dense (sdpa) baseline:        max_diff={cp_vs_dense_max:.6f}")

        BOLD = "\033[1m"
        GREEN = "\033[0;32m"
        RED = "\033[0;31m"
        RESET = "\033[0m"

        # bf16 noise floor for Qwen2.5-0.5B at logit scale ~10-20 is roughly
        # one bf16 ULP, i.e. 0.4-0.5. Threshold 1.0 leaves slack but still
        # catches the LRU-bug magnitude (>>50% drift relative to logit norms).
        ok_within = within_max < 1.0
        ok_vs_dense = cp_vs_dense_max < 1.0

        if ok_within and ok_vs_dense:
            print(
                f"  {GREEN}{BOLD}[CP-SANITY PASS] cp_size={cp_size} "
                f"dynamic-trie + Magi correct under LRU pollution{RESET}"
            )
            print(f"  {GREEN}                  and matches dense baseline within bf16 noise{RESET}")
            rc = 0
        elif not ok_within:
            print(
                f"  {RED}{BOLD}[CP-SANITY FAIL] LRU pollution affected output beyond bf16 noise "
                f"(max_diff={within_max:.4f}){RESET}"
            )
            print(f"  {RED}                  module-attribute threading likely broken.{RESET}")
            rc = 1
        else:
            print(
                f"  {RED}{BOLD}[CP-SANITY FAIL] cp_size={cp_size} dynamic-trie diverges from dense "
                f"(max_diff={cp_vs_dense_max:.4f}){RESET}"
            )
            rc = 2
    else:
        rc = 0

    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
