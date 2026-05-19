#!/usr/bin/env python3
"""Prefix-tree attention FLOP ratio calculator.

Computes theoretical speedup from prefix-tree deduplication vs standard FA3,
broken down by attention FLOPs and FFN FLOPs.

Usage:
  python3 prefix_tree_flops.py --dataset /tmp/claude/gsm8k_tree_2branch/train.parquet \
      --tokenizer /tmp/claude/MiMo-7B-RL --mbs 4

  python3 prefix_tree_flops.py --seq-len 10240 --prefix-ratio 0.60 --mbs 4 --n-leaves 4

  python3 prefix_tree_flops.py --help
"""
import argparse
from collections import defaultdict


def flops_attention(T: int, H: int, D: int, n_layers: int = 1) -> float:
    """Attention FLOPs for T tokens, H heads, D head_dim, n_layers.
    FLOPs ≈ 4 * T^2 * H * D  (QK^T + softmax approx + AV, fwd only)
    """
    return 4.0 * T * T * H * D * n_layers


def flops_ffn(T: int, hidden: int, intermediate: int, n_layers: int = 1) -> float:
    """FFN FLOPs for T tokens. For SwiGLU: 3 matmuls of (T, hidden) × (hidden, intermediate).
    FLOPs ≈ 2 * T * hidden * intermediate * 3  (×2 for fwd+bwd approx)
    """
    return 2.0 * T * hidden * intermediate * 3 * n_layers


def flat_tokens_single_level(seqs, prefix_len: int) -> int:
    """Single-level tree: [prefix | leaf_0 | leaf_1 | ...]."""
    leaf_total = sum(len(s) - prefix_len for s in seqs)
    return prefix_len + leaf_total


def flat_tokens_multilevel(seqs, prefix_len: int, turn2_groups: dict) -> int:
    """Multi-level tree: [root | turn2_A | leaf_A1 | leaf_A2 | turn2_B | ...]."""
    total = prefix_len
    for group_seqs in turn2_groups.values():
        if len(group_seqs) < 2:
            # No sharing — treat as leaves
            for s in group_seqs:
                total += len(s) - prefix_len
        else:
            # Find shared turn2 prefix within group
            from verl.utils.prefix_tree_utils import longest_common_prefix_length
            import torch
            suffixes = [torch.tensor(s[prefix_len:]) for s in group_seqs]
            t2_len = longest_common_prefix_length(suffixes)
            total += t2_len  # turn2 once
            for s in group_seqs:
                total += len(s) - prefix_len - t2_len  # leaf unique part
    return total


def compute_flop_ratio(
    seqs: list,          # list of token sequences (list of ints or tensors)
    prefix_len: int,     # shared root prefix length
    mbs: int,            # micro-batch size
    n_layers: int = 36,  # transformer layers
    n_heads: int = 32,   # attention heads
    head_dim: int = 128, # head dimension
    n_kv_heads: int = 8, # KV heads (GQA)
    hidden: int = 4096,  # hidden size
    intermediate: int = 14336,  # FFN intermediate (SwiGLU)
    multilevel: bool = False,
    turn2_groups: dict = None,
) -> dict:
    """Compute FLOP ratio: prefix-tree vs FA3."""

    n = len(seqs)
    seq_lens = [len(s) for s in seqs]
    T_full = sum(seq_lens)  # FA3: total tokens across all samples

    # Prefix-tree flat token count
    if multilevel and turn2_groups:
        T_flat = flat_tokens_multilevel(seqs, prefix_len, turn2_groups)
    else:
        T_flat = flat_tokens_single_level(seqs, prefix_len)

    prefix_sharing = (T_full - T_flat) / T_full

    # --- FA3 FLOPs ---
    # Each sample computed independently with full causal attention
    fa3_attn = sum(flops_attention(L, n_heads, head_dim, n_layers) for L in seq_lens)
    fa3_ffn  = sum(flops_ffn(L, hidden, intermediate, n_layers) for L in seq_lens)
    fa3_total = fa3_attn + fa3_ffn

    # --- Prefix-tree FLOPs ---
    # Attention: one flat computation on T_flat tokens (sparse mask)
    # Actual attention FLOPs = sum of rectangle areas × head_dim
    # For single-level: prefix causal + n × (prefix×leaf_full + leaf causal)
    # Approximate: use density × T_flat^2 as upper bound
    if multilevel and turn2_groups:
        # Multilevel: root causal + per-branch (root×branch full + branch causal)
        #             + per-leaf (root×leaf full + branch×leaf full + leaf causal)
        attn_ops = 0
        P = prefix_len
        attn_ops += P * P / 2  # root causal

        for group_seqs in turn2_groups.values():
            if len(group_seqs) < 2:
                for s in group_seqs:
                    L = len(s); leaf = L - P
                    attn_ops += P * leaf  # leaf→root full
                    attn_ops += leaf * leaf / 2  # leaf causal
            else:
                import torch
                suffixes = [torch.tensor(s[P:]) for s in group_seqs]
                from verl.utils.prefix_tree_utils import longest_common_prefix_length
                T2 = longest_common_prefix_length(suffixes)
                attn_ops += P * T2  # branch→root full
                attn_ops += T2 * T2 / 2  # branch causal
                for s in group_seqs:
                    leaf = len(s) - P - T2
                    attn_ops += P * leaf   # leaf→root full
                    attn_ops += T2 * leaf  # leaf→branch full
                    attn_ops += leaf * leaf / 2  # leaf causal
    else:
        # Single-level: prefix causal + n×(prefix→leaf + leaf causal)
        leaf_lens = [len(s) - prefix_len for s in seqs]
        attn_ops = prefix_len * prefix_len / 2  # root causal
        for L in leaf_lens:
            attn_ops += prefix_len * L   # leaf→prefix full
            attn_ops += L * L / 2        # leaf causal

    pt_attn = attn_ops * n_heads * head_dim * 4 * n_layers  # scale to FLOPs
    pt_ffn  = flops_ffn(T_flat, hidden, intermediate, n_layers)  # FFN on flat tokens
    pt_total = pt_attn + pt_ffn

    return {
        'n_samples': n,
        'seq_lens': seq_lens,
        'prefix_len': prefix_len,
        'T_full': T_full,
        'T_flat': T_flat,
        'prefix_sharing': prefix_sharing,
        'fa3_attn_gflops': fa3_attn / 1e9,
        'fa3_ffn_gflops': fa3_ffn / 1e9,
        'fa3_total_gflops': fa3_total / 1e9,
        'pt_attn_gflops': pt_attn / 1e9,
        'pt_ffn_gflops': pt_ffn / 1e9,
        'pt_total_gflops': pt_total / 1e9,
        'attn_ratio': pt_attn / fa3_attn,
        'ffn_ratio': pt_ffn / fa3_ffn,
        'total_ratio': pt_total / fa3_total,
        'speedup': fa3_total / pt_total,
    }


def print_report(r: dict, label: str = ""):
    print(f"\n{'='*60}")
    if label: print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Samples:         {r['n_samples']}")
    print(f"  Seq lengths:     {r['seq_lens']}")
    print(f"  Prefix length:   {r['prefix_len']:,} tokens")
    print(f"  Full tokens:     {r['T_full']:,}")
    print(f"  Flat tokens:     {r['T_flat']:,}  ({r['prefix_sharing']:.1%} saved)")
    print(f"")
    print(f"  {'':20s}  {'FA3':>12s}  {'PrefixTree':>12s}  {'Ratio':>8s}")
    print(f"  {'-'*56}")
    print(f"  {'Attention (GFLOPs)':20s}  {r['fa3_attn_gflops']:12.1f}  {r['pt_attn_gflops']:12.1f}  {r['attn_ratio']:8.3f}x")
    print(f"  {'FFN (GFLOPs)':20s}  {r['fa3_ffn_gflops']:12.1f}  {r['pt_ffn_gflops']:12.1f}  {r['ffn_ratio']:8.3f}x")
    print(f"  {'Total (GFLOPs)':20s}  {r['fa3_total_gflops']:12.1f}  {r['pt_total_gflops']:12.1f}  {r['total_ratio']:8.3f}x")
    print(f"")
    print(f"  Theoretical speedup: {r['speedup']:.2f}x")


def from_dataset(parquet_path: str, tokenizer_path: str, mbs: int, **model_kwargs):
    """Load a parquet dataset and compute FLOP ratio for one micro-batch."""
    import torch, pandas as pd
    from transformers import AutoTokenizer
    import sys; sys.path.insert(0, '/home/hadoop-djst-algoplat/verl-prefix-tree')
    from verl.utils.prefix_tree_magi import _resolve_multilevel_tree, _hash_prefix
    from verl.utils.prefix_tree_utils import longest_common_prefix_length

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    df = pd.read_parquet(parquet_path)
    df = df.head(mbs)  # take first micro-batch

    tokens_by_sample, prefix_segments_batch = [], []
    for idx in range(len(df)):
        msgs = df.iloc[idx]['messages']
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        tokens_by_sample.append(list(tok.encode(text)))
        cum_ids, cum = [], []
        for msg in msgs:
            msg_text = tok.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
            cum_ids.extend(tok.encode(msg_text, add_special_tokens=False))
            cum.append((_hash_prefix(torch.tensor(cum_ids)), len(cum_ids)))
        prefix_segments_batch.append(cum)

    import torch as _torch
    tbs = [_torch.tensor(t) for t in tokens_by_sample]
    root_len = longest_common_prefix_length(tbs)
    result = _resolve_multilevel_tree(tbs, prefix_segments_batch, root_len)

    multilevel = False
    turn2_groups = None
    if result is not None:
        root_l, children_info = result
        root_len = root_l
        multilevel = True
        turn2_groups = {}
        for idxs, child in children_info:
            # group sequences by branch
            turn2_groups[id(child)] = [tokens_by_sample[i] for i in idxs]

    return compute_flop_ratio(
        seqs=tokens_by_sample,
        prefix_len=root_len,
        mbs=mbs,
        multilevel=multilevel,
        turn2_groups=turn2_groups,
        **model_kwargs,
    )


def from_params(seq_len: int, prefix_ratio: float, mbs: int,
                n_leaves: int = None, **model_kwargs):
    """Compute FLOP ratio from abstract parameters."""
    prefix_len = int(seq_len * prefix_ratio)
    leaf_len = seq_len - prefix_len
    n = mbs if n_leaves is None else n_leaves
    seqs = [[0] * seq_len for _ in range(n)]
    return compute_flop_ratio(seqs=seqs, prefix_len=prefix_len, mbs=n, **model_kwargs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prefix-tree FLOP ratio calculator")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--dataset', help="Parquet dataset path")
    g.add_argument('--seq-len', type=int, help="Sequence length (abstract mode)")

    p.add_argument('--tokenizer', default='/tmp/claude/MiMo-7B-RL')
    p.add_argument('--mbs', type=int, default=4, help="Micro-batch size")
    p.add_argument('--prefix-ratio', type=float, default=0.6,
                   help="Prefix ratio (abstract mode only)")
    p.add_argument('--n-leaves', type=int, default=None,
                   help="Number of leaves (abstract mode, default=mbs)")
    # Model config
    p.add_argument('--n-layers', type=int, default=36)
    p.add_argument('--n-heads', type=int, default=32)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--n-kv-heads', type=int, default=8)
    p.add_argument('--hidden', type=int, default=4096)
    p.add_argument('--intermediate', type=int, default=14336)
    args = p.parse_args()

    model_kw = dict(n_layers=args.n_layers, n_heads=args.n_heads, head_dim=args.head_dim,
                    n_kv_heads=args.n_kv_heads, hidden=args.hidden, intermediate=args.intermediate)

    if args.dataset:
        r = from_dataset(args.dataset, args.tokenizer, args.mbs, **model_kw)
        print_report(r, f"Dataset: {args.dataset}  mbs={args.mbs}")
    else:
        r = from_params(args.seq_len, args.prefix_ratio, args.mbs, args.n_leaves, **model_kw)
        label = f"seq={args.seq_len}  prefix_ratio={args.prefix_ratio:.0%}  mbs={args.mbs}"
        print_report(r, label)


# ---------------------------------------------------------------------------
# Batch-level FLOP ratio for integration into verl step metrics
# ---------------------------------------------------------------------------

def compute_tree_flop_ratio_from_batch(data) -> float | None:
    """Compute tree FLOP ratio from a TensorDict micro-batch.

    Returns total_ratio (prefix-tree FLOPs / FA3 FLOPs) averaged across
    all micro-batches in the step, or None if not applicable.

    Uses input_ids seqlens and prefix_segments from the batch.
    Model config defaults to MiMo-7B-RL (36L, 4096H, 32/8 heads GQA).
    """
    try:
        import torch
        from verl.utils import tensordict_utils as tu
        from verl.utils.prefix_tree_magi import _resolve_multilevel_tree, _hash_prefix
        from verl.utils.prefix_tree_utils import longest_common_prefix_length

        prefix_segments = tu.get_non_tensor_data(data, 'prefix_segments', None)
        if prefix_segments is None:
            return None

        # Get sequence lengths from input_ids (NestedTensor)
        input_ids = data.get('input_ids', None)
        if input_ids is None or not input_ids.is_nested:
            return None

        seqlens = input_ids.offsets().diff().tolist()
        n = len(seqlens)
        if n < 2:
            return None

        # Build token lists from flat values for prefix detection
        flat_vals = input_ids.values().cpu()
        tokens_by_sample = []
        pos = 0
        for L in seqlens:
            L = int(L)
            tokens_by_sample.append(flat_vals[pos:pos+L])
            pos += L

        # Unwrap prefix_segments
        ps_batch = []
        for i in range(n):
            seg = prefix_segments[i]
            if hasattr(seg, 'data'):
                seg = seg.data
            ps_batch.append(list(seg) if seg is not None else [])

        # Detect sharing structure
        root_len = longest_common_prefix_length(tokens_by_sample)
        if root_len == 0:
            return None

        result = _resolve_multilevel_tree(tokens_by_sample, ps_batch, root_len)
        multilevel = False
        turn2_groups = None
        if result is not None:
            root_len, children_info = result
            multilevel = True
            turn2_groups = {
                id(child): [tokens_by_sample[i].tolist() for i in idxs]
                for idxs, child in children_info
            }

        seqs_list = [t.tolist() for t in tokens_by_sample]
        r = compute_flop_ratio(
            seqs=seqs_list,
            prefix_len=root_len,
            mbs=n,
            multilevel=multilevel,
            turn2_groups=turn2_groups,
        )
        return r['total_ratio']
    except Exception:
        return None
