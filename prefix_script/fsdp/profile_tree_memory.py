# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Profile peak GPU memory of one tree forward+backward vs one dense forward+backward.

Records a torch.cuda memory snapshot for each path. Upload the resulting
pickles to https://pytorch.org/memory_viz to see the per-tensor timeline
and identify which allocations cause the tree path to peak higher.

Output files:
  /tmp/profile_dense_memory.pickle
  /tmp/profile_tree_memory.pickle

Usage:
    USE_FSDP=1 torchrun --standalone --nproc_per_node=1 \\
        prefix_script/fsdp/profile_tree_memory.py

Env knobs (defaults match the convergence test workload):
    MODEL_PATH    HF model dir (default ~/models/Qwen/Qwen2.5-3B-Instruct)
    N             rollouts/prompt (default 8)
    P             prompt length (default 110, matches GSM8K mean)
    R             response length (default 290, matches GSM8K mean)
    USE_FSDP      1 to wrap in FSDP2 (default 1)
    ENABLE_GC     1 to enable HF gradient checkpointing (default 1, matches
                  verl training defaults). Set 0 to disable for comparison.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

N = int(os.environ.get("N", "8"))
P = int(os.environ.get("P", "110"))
R = int(os.environ.get("R", "290"))
USE_FSDP = os.environ.get("USE_FSDP", "1") == "1"
ENABLE_GC = os.environ.get("ENABLE_GC", "1") == "1"


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

    # verl training enables HF gradient checkpointing by default. Mirror it here
    # so the profile peaks line up with the production training observation.
    if ENABLE_GC:
        # FSDP2 + HF gradient_checkpointing needs use_reentrant=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[PROFILE] gradient checkpointing: ENABLED (use_reentrant=False)")
    else:
        print("[PROFILE] gradient checkpointing: DISABLED")

    for _, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if cls.endswith(("attention", "self_attn", "selfattention")):
            mod.cp_group = cp_group
    return model


def _build_batch(model, response_len: int = R):
    """1 prompt × N rollouts, prompt length P, response length response_len."""
    torch.manual_seed(42)
    vocab = model.config.vocab_size
    prompt = torch.randint(0, vocab, (P,), device="cuda")
    samples = []
    for i in range(N):
        torch.manual_seed(100 + i)
        response = torch.randint(0, vocab, (response_len,), device="cuda")
        samples.append(torch.cat([prompt, response]))
    nested = torch.nested.nested_tensor(samples, layout=torch.jagged)
    return samples, nested


def _tree_forward(model, nested, cp_group):
    from verl.models.transformers.monkey_patch import set_magi_attention_key_fsdp
    from verl.utils.prefix_tree_magi import build_prefix_tree_micro_batch, restore_flat_to_nested

    pt_batch = build_prefix_tree_micro_batch(model, nested, attention_type="magi", cp_size=1, dynamic_trie=True)
    set_magi_attention_key_fsdp(model, pt_batch.magi_key)
    flat_in = pt_batch.local_flat_input_ids.unsqueeze(0)
    flat_pos = pt_batch.local_flat_position_ids.unsqueeze(0)
    out = model(input_ids=flat_in, attention_mask=None, position_ids=flat_pos, use_cache=False)
    flat_logits = out.logits[0][: pt_batch.real_tokens]
    nested_logits = restore_flat_to_nested(flat_logits, pt_batch)
    return [nested_logits[i] for i in range(pt_batch.original_batch_size)]


def _dense_forward_per_sample(model, samples):
    """Per-sample sdpa forward (8 separate forwards, 8 retained graphs).

    Kept for reference; NOT what verl uses in training. Memory dominated by
    8× retained activations. Caller must set
    ``model.config._attn_implementation = "sdpa"`` before calling and keep
    it set through loss.backward().
    """
    return [model(input_ids=s.unsqueeze(0), attention_mask=None, use_cache=False).logits[0] for s in samples]


def _dense_forward_packed(model, samples):
    """Packed FA2 forward — matches verl's rmpad path (production dense baseline).

    Concatenate all samples into one flat (1, total_tokens) input, with
    position_ids that restart at each sample boundary. HF's flash-attention
    backend infers cu_seqlens from the position_ids resets and applies a
    per-sample causal mask without materialising a full (T, T) attention
    matrix — same effective semantics as varlen FA2.

    Caller must set ``model.config._attn_implementation = "flash_attention_2"``
    BEFORE this call AND keep it set through loss.backward() — otherwise
    gradient checkpointing recompute uses a different attention impl than
    the original forward and crashes with a saved-tensor-count mismatch.
    """
    flat_input = torch.cat(samples).unsqueeze(0)  # (1, total_tokens)
    pos_pieces = [torch.arange(s.shape[0], device="cuda", dtype=torch.long) for s in samples]
    flat_pos = torch.cat(pos_pieces).unsqueeze(0)  # (1, total_tokens), resets per sample

    out = model(
        input_ids=flat_input,
        attention_mask=None,
        position_ids=flat_pos,
        use_cache=False,
    )

    # Restore per-sample logits (so loss computation matches tree path's output shape)
    logits = out.logits[0]  # (total_tokens, vocab)
    per_sample = []
    offset = 0
    for s in samples:
        per_sample.append(logits[offset : offset + s.shape[0]])
        offset += s.shape[0]
    return per_sample


def _loss(per_sample, samples):
    losses = []
    for logits, s in zip(per_sample, samples, strict=False):
        f = logits.float()
        labels = s[1:]
        logp = torch.log_softmax(f[:-1], dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        losses.append(-logp.mean())
    return torch.stack(losses).mean()


def _magi_lru_size(cp_group) -> int:
    """Return current size of Magi's per-cp_group runtime LRU. 0 if Magi
    internals not importable."""
    try:
        from magi_attention.api.magi_attn_interface import dist_attn_runtime_dict_mgr

        return len(dist_attn_runtime_dict_mgr.keys(cp_group))
    except Exception:
        return 0


def _magi_lru_clear(cp_group):
    """Clear Magi's per-cp_group LRU."""
    try:
        from magi_attention.api.magi_attn_interface import clear_cache

        clear_cache(cp_group)
    except Exception as e:
        print(f"[MULTI] WARN: clear_cache failed: {e}")


def _profile_multi_forward(
    label: str,
    model,
    cp_group,
    forward_kind: str,  # "tree" or "dense_packed"
    n_iters: int,
    response_lengths: list[int],
    clear_lru: bool = False,
):
    """Run n_iters forward+backward with varying response lengths to expose
    LRU / persistent-state memory accumulation. Reports peak memory growth
    across iterations + Magi LRU size after each iter.

    forward_kind:
      "tree":          builds a fresh Magi key per iter (registers in Magi LRU)
      "dense_packed":  FA2 packed forward, no persistent state

    clear_lru: when True, call magi_attention.api.clear_cache after each iter
        to test whether LRU accumulation drives the peak growth.
    """
    torch.cuda.empty_cache()
    if forward_kind == "tree":
        _magi_lru_clear(cp_group)  # start fresh

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 1e9
    print(f"[MULTI] {label}: baseline allocated = {base:.2f} GB,  clear_lru = {clear_lru}")
    print(
        f"[MULTI] {label}: running {n_iters} iters, response lengths = "
        f"[{min(response_lengths)}..{max(response_lengths)}]"
    )
    print(f"[MULTI] {'iter':>5} {'resp_len':>10} {'peak_GB':>10} {'final_GB':>10} {'delta_GB':>10} {'magi_lru':>10}")
    print("-" * 70)

    peaks: list[float] = []
    finals: list[float] = []
    for i, r_len in enumerate(response_lengths):
        samples, nested = _build_batch(model, response_len=r_len)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        if forward_kind == "tree":
            per_sample = _tree_forward(model, nested, cp_group)
        elif forward_kind == "dense_packed":
            per_sample = _dense_forward_packed(model, samples)
        else:
            raise ValueError(f"unknown forward_kind={forward_kind}")
        loss = _loss(per_sample, samples)
        loss.backward()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() / 1e9
        final = torch.cuda.memory_allocated() / 1e9
        lru_size = _magi_lru_size(cp_group)
        peaks.append(peak)
        finals.append(final)
        delta = (peaks[i] - peaks[0]) if i > 0 else 0.0
        print(f"[MULTI] {i:>5} {r_len:>10} {peak:>10.2f} {final:>10.2f} {delta:>+10.2f} {lru_size:>10}")

        if clear_lru and forward_kind == "tree":
            _magi_lru_clear(cp_group)
            torch.cuda.empty_cache()

    print(
        f"[MULTI] {label}: peak growth iter[0]→iter[-1] = "
        f"{peaks[-1] - peaks[0]:+.2f} GB ({peaks[0]:.2f} → {peaks[-1]:.2f})"
    )
    return peaks, finals


def _profile_one(label: str, fwd_fn, samples, output_path: str):
    """Record one forward+backward with memory history → dump snapshot pickle."""
    # Reset baseline so peak is for THIS forward only
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 1e9
    print(f"[PROFILE] {label}: baseline allocated = {base:.2f} GB")

    # Start recording history (each allocation + free with stack)
    torch.cuda.memory._record_memory_history(max_entries=200_000)

    per_sample = fwd_fn()
    loss = _loss(per_sample, samples)
    loss.backward()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    final = torch.cuda.memory_allocated() / 1e9
    print(f"[PROFILE] {label}: peak={peak:.2f} GB  final={final:.2f} GB  (loss={loss.item():.4f})")

    # Dump snapshot for memory_viz
    torch.cuda.memory._dump_snapshot(output_path)
    print(f"[PROFILE] {label}: snapshot → {output_path}")
    torch.cuda.memory._record_memory_history(enabled=None)


def main() -> int:
    _init_dist()
    cp_group = dist.group.WORLD

    model_path = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/Qwen/Qwen2.5-3B-Instruct"))
    print(f"[PROFILE] USE_FSDP={USE_FSDP}  N={N} P={P} R={R}")
    print(f"[PROFILE] Loading {model_path}")
    model = _load_model(model_path, cp_group)
    samples, nested = _build_batch(model)
    print(f"[PROFILE] total_tokens = {sum(s.shape[0] for s in samples)}, prefix_shared = {P * (N - 1)}")
    print()

    # ── Multi-forward LRU-accumulation test (env: MULTI_N) — if set, runs that
    # many forward+backward iters with varying response lengths to expose any
    # persistent state (Magi LRU mgr cache, FSDP buffers, etc.) that grows
    # across iters. Mimics what a real training step does (~96 forwards
    # over 32 distinct micro-batch shapes).
    multi_n = int(os.environ.get("MULTI_N", "0"))
    if multi_n > 0:
        # Spread response lengths: 100..700 in N steps (matches GSM8K's
        # observed response_length min/max range from R2_tree run)
        lo, hi = 100, 700
        if multi_n == 1:
            response_lengths = [R]
        else:
            step = (hi - lo) // (multi_n - 1)
            response_lengths = [lo + step * i for i in range(multi_n)]

        # ── A: tree path with LRU left intact (method 1 — measure accumulation)
        print("=" * 70)
        print(f"[PROFILE] A: tree path, {multi_n} iters, LRU INTACT (measures accumulation)")
        print("=" * 70)
        model.config._attn_implementation = "Magi_Attention"
        peaks_tree_keep, _ = _profile_multi_forward(
            "tree_lru_keep", model, cp_group, "tree", multi_n, response_lengths, clear_lru=False
        )
        print()

        # ── B: tree path with LRU cleared after each iter (method 3 — test the fix)
        print("=" * 70)
        print(f"[PROFILE] B: tree path, {multi_n} iters, LRU CLEARED after each iter")
        print("=" * 70)
        model.config._attn_implementation = "Magi_Attention"
        peaks_tree_clear, _ = _profile_multi_forward(
            "tree_lru_clear", model, cp_group, "tree", multi_n, response_lengths, clear_lru=True
        )
        print()

        # ── C: dense packed baseline
        print("=" * 70)
        print(f"[PROFILE] C: dense packed path, {multi_n} iters (no LRU, sanity baseline)")
        print("=" * 70)
        model.config._attn_implementation = "flash_attention_2"
        peaks_dense, _ = _profile_multi_forward(
            "dense", model, cp_group, "dense_packed", multi_n, response_lengths, clear_lru=False
        )
        print()

        # ── Summary table
        print("=" * 70)
        print("[PROFILE] SUMMARY")
        print("=" * 70)
        print(f"{'condition':<25} {'iter[0] peak':>15} {'iter[-1] peak':>15} {'growth':>10}")
        print("-" * 70)
        for cond, peaks in [
            ("tree (LRU intact)", peaks_tree_keep),
            ("tree (LRU cleared)", peaks_tree_clear),
            ("dense packed", peaks_dense),
        ]:
            growth = peaks[-1] - peaks[0]
            print(f"{cond:<25} {peaks[0]:>15.2f} {peaks[-1]:>15.2f} {growth:>+10.2f}")
        print()
        print("[PROFILE] Hypothesis check:")
        print("  - LRU accumulation drives peak ⇔ tree(intact).growth >> tree(cleared).growth")
        print("  - dense.growth should ≈ 0 (no persistent state)")

        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    # ── Tree path
    print("=" * 70)
    print("[PROFILE] Tree (dynamic-trie + Magi) forward + backward")
    print("=" * 70)
    model.zero_grad(set_to_none=True)
    model.config._attn_implementation = "Magi_Attention"
    _profile_one(
        "tree",
        lambda: _tree_forward(model, nested, cp_group),
        samples,
        "/tmp/profile_tree_memory.pickle",
    )
    print()

    # ── Dense packed (FA2) — what verl ACTUALLY uses in training
    print("=" * 70)
    print("[PROFILE] Dense packed (FA2 rmpad) forward + backward — production-equivalent")
    print("=" * 70)
    model.zero_grad(set_to_none=True)
    # Set FA2 BEFORE forward and keep through backward — grad ckpt recompute
    # must see the same _attn_implementation as the original forward.
    model.config._attn_implementation = "flash_attention_2"
    _profile_one(
        "dense_packed",
        lambda: _dense_forward_packed(model, samples),
        samples,
        "/tmp/profile_dense_packed_memory.pickle",
    )
    print()

    # ── Dense per-sample (sdpa) — kept as reference, NOT production
    if os.environ.get("ALSO_DENSE_PER_SAMPLE", "0") == "1":
        print("=" * 70)
        print("[PROFILE] Dense per-sample sdpa forward + backward (reference, NOT production)")
        print("=" * 70)
        model.zero_grad(set_to_none=True)
        model.config._attn_implementation = "sdpa"
        _profile_one(
            "dense_per_sample",
            lambda: _dense_forward_per_sample(model, samples),
            samples,
            "/tmp/profile_dense_per_sample_memory.pickle",
        )

    print()
    print("=" * 70)
    print("[PROFILE] DONE. Upload to https://pytorch.org/memory_viz :")
    print("           /tmp/profile_tree_memory.pickle         (tree packed Magi)")
    print("           /tmp/profile_dense_packed_memory.pickle (production-equivalent dense)")
    if os.environ.get("ALSO_DENSE_PER_SAMPLE", "0") == "1":
        print("           /tmp/profile_dense_per_sample_memory.pickle (reference, per-sample sdpa)")
    print("=" * 70)
    print(
        "[PROFILE] NOTE: tree-vs-dense_packed is the comparison that matches the "
        "8×H100 training observation (16.6 vs 11.7 GB per rank)."
    )

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
