"""CPU test: PrefixSubTrie pickle round-trip preserves layout/leaf mapping
and reconstructs children pointers from flat_idx after unpickling."""
import pickle

import torch

from verl.utils.prefix_tree.dynamic import greedy_build_tries, subtrie_view
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _make_subtrie(raw_seqs, keep_ids):
    tries, _ = greedy_build_tries(raw_seqs)
    subtrie = subtrie_view(tries[0], set(keep_ids))
    assert subtrie is not None
    return subtrie


def _build_params(subtrie, samples):
    lm = [torch.ones(len(s), dtype=torch.float32) for s in samples]
    return build_layout_from_tree_node(samples, subtrie, loss_masks_by_sample=lm)


def _samples(raw):
    return [torch.tensor(s, dtype=torch.long) for s in raw]


def test_basic_and_duplicates_round_trip():
    raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    st = _make_subtrie(raw, [0, 1, 2])
    samps = _samples(raw)
    p1 = _build_params(st, samps)
    st2 = pickle.loads(pickle.dumps(st))
    p2 = _build_params(st2, samps)
    assert torch.equal(p1.tree_packed_tokens, p2.tree_packed_tokens)
    assert p1.leaf_to_sample == p2.leaf_to_sample
    assert p1.q_ranges == p2.q_ranges
    assert p1.prefix_range == p2.prefix_range
    # Duplicates case: set-equality on leaf_to_sample (order may differ)
    dup = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 5, 6]]
    st_dup = _make_subtrie(dup, [0, 1, 2])
    st_dup2 = pickle.loads(pickle.dumps(st_dup))
    p_d1 = _build_params(st_dup, _samples(dup))
    p_d2 = _build_params(st_dup2, _samples(dup))
    assert torch.equal(p_d1.tree_packed_tokens, p_d2.tree_packed_tokens)
    assert set(p_d1.leaf_to_sample) == set(p_d2.leaf_to_sample)


def test_children_reconstructed_after_unpickling():
    raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    st2 = pickle.loads(pickle.dumps(_make_subtrie(raw, [0, 1, 2])))
    valid = {n.node_idx for n in st2.nodes}
    children = [c for c in st2.nodes[0].children.values() if c.node_idx in valid]
    assert len(children) > 0
