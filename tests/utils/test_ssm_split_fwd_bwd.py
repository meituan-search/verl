"""
Unit test: SSM split forward and backward correctness.

Verifies that for sequences AB and AC sharing prefix A:
  1. FWD: split(A → s_A, then B/C from s_A) == full(A+B) / full(A+C)
  2. BWD: gradients on A's inputs from split(B_loss + C_loss)
          match gradients from full(A+B_loss) + full(A+C_loss)

Tests both kernels used by Megatron GDN:
  - chunk_gated_delta_rule (recurrent state)
  - causal_conv1d_fn       (conv state)

Run: python3 test_ssm_split_fwd_bwd.py
"""

import torch
from torch.testing import assert_close

DEVICE = "cuda"
DTYPE  = torch.float32   # bf16 too imprecise for grad checks
CHUNK  = 64              # chunk_gated_delta_rule chunk_size
ATOL   = 1e-4


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def leaf(*shape, scale=0.1):
    return (torch.randn(*shape, dtype=DTYPE, device=DEVICE) * scale).requires_grad_(True)

def leaf_g(*shape):
    """Decay: must be ≤ 0 (log-space). Positive g → exp overflow → NaN."""
    return (-torch.rand(*shape, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

def leaf_beta(*shape):
    """Beta in (0,1) as a leaf (no grad_fn from sigmoid)."""
    return torch.sigmoid(torch.randn(*shape, dtype=DTYPE, device=DEVICE)).requires_grad_(True)

def cl(*shape):
    """Channel-last (B,D,L) tensor: stride(1)==1, required by causal_conv1d for final_states."""
    B, D, L = shape
    return torch.randn(B, L, D, dtype=DTYPE, device=DEVICE).transpose(1, 2).requires_grad_(True)


# ─────────────────────────────────────────────────────────────
# 1. GDN — forward
# ─────────────────────────────────────────────────────────────

def test_gdn_split_forward():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, H, K, V = 1, 4, 32, 32
    LA, LB = CHUNK, CHUNK   # split at exact chunk boundary

    torch.manual_seed(0)
    q, k, v, g, beta = [leaf(B, LA+LB, H, K) if i < 3 else
                         leaf_g(B, LA+LB, H) if i == 3 else
                         leaf_beta(B, LA+LB, H) for i in range(5)]

    out_full, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                          initial_state=None, output_final_state=False)

    _, s_A = chunk_gated_delta_rule(q[:, :LA], k[:, :LA], v[:, :LA],
                                     g=g[:, :LA], beta=beta[:, :LA],
                                     initial_state=None, output_final_state=True)
    out_B, _ = chunk_gated_delta_rule(q[:, LA:], k[:, LA:], v[:, LA:],
                                       g=g[:, LA:], beta=beta[:, LA:],
                                       initial_state=s_A, output_final_state=False)

    assert_close(out_B, out_full[:, LA:], atol=ATOL, rtol=0)
    print("  ✓ GDN fwd: split at chunk boundary == full")


# ─────────────────────────────────────────────────────────────
# 2. GDN — backward: B and C share A, gradients accumulate correctly
# ─────────────────────────────────────────────────────────────

def test_gdn_split_backward():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, H, K, V = 1, 2, 32, 32
    LA, LB, LC = CHUNK, CHUNK, CHUNK

    torch.manual_seed(1)
    qA  = leaf(B, LA, H, K);  kA  = leaf(B, LA, H, K)
    vA  = leaf(B, LA, H, V);  gA  = leaf_g(B, LA, H);  bA = leaf_beta(B, LA, H)
    qB  = leaf(B, LB, H, K);  kB  = leaf(B, LB, H, K)
    vB  = leaf(B, LB, H, V);  gB  = leaf_g(B, LB, H);  bB = leaf_beta(B, LB, H)
    qC  = leaf(B, LC, H, K);  kC  = leaf(B, LC, H, K)
    vC  = leaf(B, LC, H, V);  gC  = leaf_g(B, LC, H);  bC = leaf_beta(B, LC, H)

    # Reference: two independent full runs through A
    def full_run(qX, kX, vX, gX, bX):
        q = torch.cat([qA, qX], 1); k = torch.cat([kA, kX], 1)
        v = torch.cat([vA, vX], 1); g = torch.cat([gA, gX], 1); b = torch.cat([bA, bX], 1)
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=b,
                                         initial_state=None, output_final_state=False)
        return out[:, LA:].sum()

    (full_run(qB, kB, vB, gB, bB) + full_run(qC, kC, vC, gC, bC)).backward()
    ref = {n: t.grad.clone() for n, t in [('qA',qA),('kA',kA),('vA',vA),('gA',gA)]}
    for t in [qA, kA, vA, gA, bA]: t.grad.zero_()

    # Split: A once → s_A leaf → B and C accumulate grad → chain to A
    _, s_A_graph = chunk_gated_delta_rule(qA, kA, vA, g=gA, beta=bA,
                                           initial_state=None, output_final_state=True)
    s_A = s_A_graph.detach().requires_grad_(True)  # leaf for diamond DAG

    out_B, _ = chunk_gated_delta_rule(qB, kB, vB, g=gB, beta=bB,
                                       initial_state=s_A, output_final_state=False)
    out_C, _ = chunk_gated_delta_rule(qC, kC, vC, g=gC, beta=bC,
                                       initial_state=s_A, output_final_state=False)
    (out_B.sum() + out_C.sum()).backward()   # accumulates s_A.grad from B+C
    s_A_graph.backward(s_A.grad)             # chain back through A

    for name, t in [('qA',qA),('kA',kA),('vA',vA),('gA',gA)]:
        assert_close(t.grad, ref[name], atol=ATOL, rtol=0,
                     msg=f"GDN bwd {name} mismatch")
    print("  ✓ GDN bwd: split AB+AC grads match full reference")


# ─────────────────────────────────────────────────────────────
# 3. Conv — forward
# ─────────────────────────────────────────────────────────────

def test_conv_split_forward():
    from causal_conv1d import causal_conv1d_fn
    B, D, LA, LB, W = 1, 64, 32, 32, 4

    torch.manual_seed(2)
    # (B,L,D).T → (B,D,L) with stride(1)==1 (channel-last required)
    x = torch.randn(B, LA+LB, D, dtype=DTYPE, device=DEVICE).transpose(1, 2)
    w = torch.randn(D, W, dtype=DTYPE, device=DEVICE)

    out_full = causal_conv1d_fn(x, weight=w, activation="silu")

    xA = x[:, :, :LA]; xB = x[:, :, LA:]
    out_A, s_A = causal_conv1d_fn(xA, weight=w, activation="silu", return_final_states=True)
    out_B, _   = causal_conv1d_fn(xB, weight=w, activation="silu",
                                   initial_states=s_A, return_final_states=True)

    assert_close(out_B, out_full[:, :, LA:], atol=ATOL, rtol=0)
    print("  ✓ Conv fwd: split == full")


# ─────────────────────────────────────────────────────────────
# 4. Conv — backward
# ─────────────────────────────────────────────────────────────

def test_conv_split_backward():
    from causal_conv1d import causal_conv1d_fn
    B, D, LA, LB, LC, W = 1, 32, 32, 16, 16, 4

    torch.manual_seed(3)
    xA = cl(B, D, LA); xB = cl(B, D, LB); xC = cl(B, D, LC)
    w  = torch.randn(D, W, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def full_conv(xX):
        out = causal_conv1d_fn(torch.cat([xA, xX], dim=2), weight=w, activation="silu")
        return out[:, :, LA:].sum()

    (full_conv(xB) + full_conv(xC)).backward()
    ref_xA = xA.grad.clone(); ref_w = w.grad.clone()
    xA.grad.zero_(); w.grad.zero_()

    # Split: A → state (leaf) → B+C accumulate → chain to A
    _, s_A_graph = causal_conv1d_fn(xA, weight=w, activation="silu", return_final_states=True)
    s_A = s_A_graph.detach().requires_grad_(True)

    outB, _ = causal_conv1d_fn(xB, weight=w, activation="silu",
                                initial_states=s_A, return_final_states=True)
    outC, _ = causal_conv1d_fn(xC, weight=w, activation="silu",
                                initial_states=s_A, return_final_states=True)
    (outB.sum() + outC.sum()).backward()
    s_A_graph.backward(s_A.grad)

    assert_close(xA.grad, ref_xA, atol=ATOL, rtol=0, msg="Conv bwd xA mismatch")
    assert_close(w.grad,  ref_w,  atol=ATOL, rtol=0, msg="Conv bwd w mismatch")
    print("  ✓ Conv bwd: split AB+AC grads match full reference")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── GDN (chunk_gated_delta_rule) ──")
    test_gdn_split_forward()
    test_gdn_split_backward()

    print("\n── Conv (causal_conv1d) ──")
    test_conv_split_forward()
    test_conv_split_backward()

    print("\nAll tests passed.")
