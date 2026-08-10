"""CPU tests for verl/utils/prefix_tree/magi.py: flat layout build,
restore_flat_to_nested round-trip, loss-mask flattening, and the
build_prefix_tree_micro_batch integration entrypoint. The MAGI-key
construction itself requires GPU + distributed and is not covered here.

Note: the layout uses a boundary-donation mechanism (non-leaf nodes donate
their last token to each child), so packed tokens include repeated prefix
tokens. Tests assert structural invariants rather than exact token sequences.
"""
from __future__ import annotations

import torch

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.magi import PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _build_params(tokens):
    result = build_tree_dynamic(tokens)
    assert result is not None, "Expected shared prefix trie"
    return build_layout_from_tree_node(tokens, result), result


def _build_pt_batch(tokens):
    from verl.utils.prefix_tree.magi import PackRestorationParam

    params, subtrie = _build_params(tokens)
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
        ),
        subtrie=subtrie,
    )


def _build_pt_batch(tokens):
    from verl.utils.prefix_tree.magi import PackRestorationParam

    params, subtrie = _build_params(tokens)
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
        ),
        subtrie=subtrie,
    )


def test_basic_shared_prefix_flat_layout_and_flex_rects():
    tokens = [
        torch.tensor([10, 20, 30, 41, 42]),
        torch.tensor([10, 20, 30, 51]),
        torch.tensor([10, 20, 30, 61, 62, 63]),
    ]
    params, _ = _build_params(tokens)
    # Packed starts with the shared prefix [10,20,30]; prefix_range = retained
    # prefix after donation (last token donated to children).
    assert list(params.tree_packed_tokens[:3].tolist()) == [10, 20, 30]
    assert params.prefix_range[0] == 0 and params.prefix_range[1] >= 1
    assert len(params.leaf_ranges) == 3  # one per sample
    assert params.total_seqlen_q >= max(t.numel() for t in tokens)
    # Flex rectangles: causal self-rects + full rects (leaves attending to prefix)
    short = [torch.tensor([10, 20, 30, 41, 42]), torch.tensor([10, 20, 30, 51])]
    sp, _ = _build_params(short)
    rects = set(zip(sp.q_ranges, sp.k_ranges, sp.mask_types, strict=False))
    assert any(m == "causal" for _, _, m in rects)
    assert any(m == "full" for _, _, m in rects)


def test_restore_token_ids_round_trip():
    tokens = [
        torch.tensor([10, 20, 30, 41, 42]),
        torch.tensor([10, 20, 30, 51]),
        torch.tensor([10, 20, 30, 61, 62, 63]),
    ]
    pt_batch = _build_pt_batch(tokens)
    restored = restore_flat_to_nested(pt_batch.tree_packed_input_ids, pt_batch)
    offsets, vals = restored.offsets(), restored.values()
    lengths = offsets.diff().tolist()
    assert lengths == [5, 4, 6]
    pos = 0
    for i, orig in enumerate(tokens):
        assert torch.equal(vals[pos : pos + int(lengths[i])], orig), f"sample {i} mismatch"
        pos += int(lengths[i])


def test_loss_mask_flattened():
    tokens = [torch.tensor([10, 20, 30, 41, 42]), torch.tensor([10, 20, 30, 51])]
    loss_masks = [torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])]
    subtrie = build_tree_dynamic(tokens)
    assert subtrie is not None
    params = build_layout_from_tree_node(tokens, subtrie, loss_masks_by_sample=loss_masks)
    lm = params.tree_packed_loss_mask
    assert lm is not None
    # Packed mask length matches packed tokens; prefix donations carry their own mask
    assert lm.shape[0] == params.tree_packed_tokens.shape[0]
    assert int(lm[0]) == 0 and int(lm[1]) == 0  # prefix tokens [10,20] are loss-masked-out


def test_build_prefix_tree_micro_batch_unpacks_nested(monkeypatch):
    """Integration: NestedTensor input -> flat layout via build_prefix_tree_micro_batch.
    magi_attention is stubbed by conftest; _build_magi_key is monkeypatched out.

    Skipped when the full verl dependency stack (codetiming, etc.) isn't
    importable — forward.py transitively imports verl.workers.config which
    needs real PyPI packages the CPU test env may lack.
    """
    import types

    pytest = __import__("pytest")
    pytest.importorskip("codetiming")
    import verl.utils.prefix_tree.forward as ptf
    import verl.utils.prefix_tree.magi as ptm

    monkeypatch.setattr(ptf, "_build_magi_key", lambda model, params: object())
    cfg = types.SimpleNamespace(num_attention_heads=8, num_query_groups=8, kv_channels=128, fp8=None)
    model = types.SimpleNamespace(config=cfg, pre_process=True, post_process=True)
    tensors = [torch.tensor(t) for t in [[10, 20, 30, 41, 42], [10, 20, 30, 51], [10, 20, 30, 61, 62, 63]]]
    input_ids = torch.nested.nested_tensor(tensors, layout=torch.jagged)
    result = ptm.build_prefix_tree_micro_batch(model, input_ids)
    assert result is not None and len(result.restoration.segment_ranges) == 3
    # Packed starts with the shared prefix [10,20,30]
    assert list(result.tree_packed_input_ids[:3].tolist()) == [10, 20, 30]
