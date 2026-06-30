"""CPU unit test: PrefixSubTrie pickle round-trip."""
import pickle, sys, os
sys.path.insert(0, os.getcwd())

import torch
from verl.utils.prefix_tree.dynamic import greedy_build_tries, subtrie_view
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def make_subtrie(raw_seqs, keep_ids):
    tries, _ = greedy_build_tries(raw_seqs, max_tokens_per_tree=10000)
    subtrie = subtrie_view(tries[0], set(keep_ids))
    assert subtrie is not None
    return subtrie


def build_params(subtrie, samples):
    lm = [torch.ones(len(s), dtype=torch.float32) for s in samples]
    return build_layout_from_tree_node(samples, subtrie, loss_masks_by_sample=lm)


def test_basic():
    raw = [[1,2,3,4],[1,2,3,5],[1,2,6,7]]
    st = make_subtrie(raw, [0,1,2])
    samps = [torch.tensor(s, dtype=torch.long) for s in raw]
    p1 = build_params(st, samps)
    blob = pickle.dumps(st)
    print(f"  pickle size: {len(blob):,} bytes")
    st2 = pickle.loads(blob)
    p2 = build_params(st2, samps)
    assert torch.equal(p1.tree_packed_tokens, p2.tree_packed_tokens), "tokens mismatch"
    assert p1.leaf_to_sample == p2.leaf_to_sample, "leaf_to_sample mismatch"
    assert p1.q_ranges == p2.q_ranges, "q_ranges mismatch"
    assert p1.prefix_range == p2.prefix_range, "prefix_range mismatch"
    print("  PASS: basic round-trip")


def test_size():
    # Long prompt + per-sample responses to simulate production scale
    prompt = list(range(1000))
    raw = [prompt + list(range(5000+i*50, 5000+i*50+200)) for i in range(8)]
    tries, _ = greedy_build_tries(raw, max_tokens_per_tree=1_000_000)
    full_size = len(pickle.dumps(tries[0]))
    sub = subtrie_view(tries[0], {0,1,2,3})
    sub_size = len(pickle.dumps(sub))
    ratio = sub_size / full_size
    print(f"  full trie: {full_size:,} bytes  subtrie(4/8): {sub_size:,} bytes  ratio={ratio:.2f}")
    assert sub_size < full_size, f"subtrie ({sub_size}) should be < full trie ({full_size})"
    print(f"  PASS: subtrie is {1-ratio:.0%} smaller than full trie")


def test_duplicates():
    raw = [[1,2,3,4],[1,2,3,4],[1,2,5,6]]
    st = make_subtrie(raw, [0,1,2])
    samps = [torch.tensor(s, dtype=torch.long) for s in raw]
    p1 = build_params(st, samps)
    p2 = build_params(pickle.loads(pickle.dumps(st)), samps)
    assert torch.equal(p1.tree_packed_tokens, p2.tree_packed_tokens), "dup tokens mismatch"
    assert set(p1.leaf_to_sample) == set(p2.leaf_to_sample)
    print("  PASS: duplicate sequences round-trip")


def test_children_reconstructed():
    raw = [[1,2,3,4],[1,2,3,5],[1,2,6,7]]
    st2 = pickle.loads(pickle.dumps(make_subtrie(raw, [0,1,2])))
    valid = {n.flat_idx for n in st2.nodes}
    children = [c for c in st2.nodes[0].children.values() if c.flat_idx in valid]
    assert len(children) > 0, "no children after unpickling"
    print(f"  root has {len(children)} subtrie child(ren) after unpickling")
    print("  PASS: children reconstructed")


if __name__ == "__main__":
    for fn in [test_basic, test_size, test_duplicates, test_children_reconstructed]:
        print(f"{fn.__name__}:")
        fn()
    print("\nAll tests passed.")
