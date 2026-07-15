# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for prepare_micro_batches in verl/workers/engine/utils.py.

Tests verify that all three code paths (dynamic+prefix, dynamic only, fixed)
attach prefix_tree_subtree when use_prefix_tree=True, that micro-batches have
correct subtrees matching their samples, and that non-tensor segment arrays are
sliced correctly per micro-batch.

Import strategy
---------------
``verl/workers/engine/__init__.py`` eagerly imports FSDP/Megatron engines that
need ``peft``, ``torch.distributed``, etc.  ``verl/workers/engine/utils.py``
itself imports ``DatasetPadMode`` which goes through ``verl.utils.dataset.__init__``
and pulls in the HuggingFace ``datasets`` library.  Neither of these heavy
dependencies is available in a CPU-only unit-test environment.

We work around both by:
1. Inserting a tiny stub module for ``datasets`` before any verl import, so that
   ``verl.utils.dataset.rl_dataset`` (and friends) can be loaded without errors.
2. Inserting a stub ``verl.utils.dataset`` package that exposes only
   ``DatasetPadMode`` (from ``dataset_utils``) — the sole symbol needed by
   ``utils.py``.
3. Loading ``verl.workers.engine.utils`` directly from its file path via
   ``importlib`` so the package ``__init__`` (with its FSDP imports) is never
   executed.
"""

import importlib.util
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from tensordict import TensorDict

# ---------------------------------------------------------------------------
# Step 1 — load verl.utils.dataset.dataset_utils directly (avoids
#           triggering the package __init__ which imports HuggingFace datasets)
# ---------------------------------------------------------------------------
import os as _os  # noqa: E402

_DATASET_UTILS_PATH = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(__file__),
        "..", "..", "..", "verl", "utils", "dataset", "dataset_utils.py",
    )
)
_dsu_spec = importlib.util.spec_from_file_location("verl.utils.dataset.dataset_utils", _DATASET_UTILS_PATH)
_dsu_mod = importlib.util.module_from_spec(_dsu_spec)
sys.modules["verl.utils.dataset.dataset_utils"] = _dsu_mod
_dsu_spec.loader.exec_module(_dsu_mod)

# Stub verl.utils.dataset package so that imports of DatasetPadMode work without
# pulling in rl_dataset.py (which needs the HuggingFace `datasets` library).
_dataset_pkg = types.ModuleType("verl.utils.dataset")
_dataset_pkg.DatasetPadMode = _dsu_mod.DatasetPadMode
_dataset_pkg.dataset_utils = _dsu_mod
sys.modules["verl.utils.dataset"] = _dataset_pkg

# ---------------------------------------------------------------------------
# Step 2 — load verl.workers.engine.utils directly (bypass package __init__)
# ---------------------------------------------------------------------------

_UTILS_PATH = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(__file__),
        "..", "..", "..", "verl", "workers", "engine", "utils.py",
    )
)
_spec = importlib.util.spec_from_file_location("verl.workers.engine._utils_direct", _UTILS_PATH)
_engine_utils_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _engine_utils_mod
_spec.loader.exec_module(_engine_utils_mod)

prepare_micro_batches = _engine_utils_mod.prepare_micro_batches

# Now it is safe to import verl.utils.tensordict_utils.
from verl.utils import tensordict_utils as tu  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    batch_size: int,
    seq_len: int,
    use_dynamic_bsz: bool = True,
    use_prefix_tree: bool = False,
    max_token_len_per_gpu: int = 512,
    micro_batch_size_per_gpu: int | None = None,
    force_group_size: int = 1,
) -> TensorDict:
    """Build a minimal TensorDict that prepare_micro_batches can consume.

    All sequences are identical (arange) so the trie finds a full shared prefix.
    """
    input_ids_list = [torch.arange(seq_len, dtype=torch.long) for _ in range(batch_size)]
    input_ids = torch.nested.as_nested_tensor(input_ids_list, layout=torch.jagged)

    td = TensorDict(
        {"input_ids": input_ids},
        batch_size=[batch_size],
    )
    # max_token_len_per_gpu is a scalar metadata field — store as NonTensorData, not a tensor,
    # because TensorDict with batch_size=[N] rejects 0-d tensors.
    tu.assign_non_tensor(td, max_token_len_per_gpu=max_token_len_per_gpu)
    tu.assign_non_tensor(td, use_dynamic_bsz=use_dynamic_bsz)
    tu.assign_non_tensor(td, use_prefix_tree=use_prefix_tree)
    tu.assign_non_tensor(td, sp_size=1)
    tu.assign_non_tensor(td, force_group_size=force_group_size)

    if not use_dynamic_bsz and micro_batch_size_per_gpu is not None:
        tu.assign_non_tensor(td, micro_batch_size_per_gpu=micro_batch_size_per_gpu)
        td["micro_batch_size_per_gpu"] = torch.tensor(micro_batch_size_per_gpu)

    return td


# ---------------------------------------------------------------------------
# Tests: Path 1 — dynamic_bsz=True AND prefix_tree=True
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesDynamicPrefixTree(unittest.TestCase):
    """Path 1: use_dynamic_bsz=True AND use_prefix_tree=True."""

    def _run(self, batch_size=4, seq_len=8):
        td = _make_data(
            batch_size=batch_size,
            seq_len=seq_len,
            use_dynamic_bsz=True,
            use_prefix_tree=True,
            max_token_len_per_gpu=512,
        )
        return prepare_micro_batches(td)

    def test_returns_list_of_tensordicts(self):
        micro_batches, _ = self._run()
        self.assertIsInstance(micro_batches, list)
        self.assertGreater(len(micro_batches), 0)
        for mb in micro_batches:
            self.assertIsInstance(mb, TensorDict)

    def test_all_microbatches_have_prefix_tree_subtree(self):
        """All micro-batches must carry prefix_tree_subtree when use_prefix_tree=True."""
        micro_batches, _ = self._run()
        for i, mb in enumerate(micro_batches):
            subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
            self.assertIsNotNone(subtree, f"micro-batch {i} missing prefix_tree_subtree")

    def test_batch_idx_list_covers_all_samples(self):
        batch_size = 4
        _, batch_idx_list = self._run(batch_size=batch_size)
        all_indices = set()
        for idx_group in batch_idx_list:
            all_indices.update(idx_group)
        self.assertEqual(all_indices, set(range(batch_size)))

    def test_subtree_sample_count_matches_microbatch(self):
        """Each subtree's batch_size must match the number of samples in that micro-batch."""
        micro_batches, batch_idx_list = self._run(batch_size=4)
        for idx_group, mb in zip(batch_idx_list, micro_batches, strict=False):
            subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
            if subtree is not None:
                self.assertEqual(subtree.batch_size, len(idx_group))

    def test_trie_built_once_reused_per_microbatch(self):
        """All micro-batch subtrees must share a single PrefixTrie source object.

        The trie is built once and stored in data, then passed to
        prepare_prefix_tree_micro_batches which creates PrefixSubTrie per micro-batch.
        Each subtree carries a ``source`` back-reference to the PrefixTrie.  All
        having the same ``id(source)`` proves the trie was constructed exactly once.
        """
        td = _make_data(batch_size=4, seq_len=8, use_dynamic_bsz=True, use_prefix_tree=True, max_token_len_per_gpu=512)
        micro_batches, _ = prepare_micro_batches(td)
        sources = {id(tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None).source)
                   for mb in micro_batches
                   if tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None) is not None}
        self.assertEqual(len(sources), 1, "trie was rebuilt per micro-batch — multiple source objects found")


# ---------------------------------------------------------------------------
# Tests: Path 2 — dynamic_bsz=True, prefix_tree=False
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesDynamicOnly(unittest.TestCase):
    """Path 2: use_dynamic_bsz=True, use_prefix_tree=False."""

    def _run(self, batch_size=4, seq_len=8, max_token_len=512):
        td = _make_data(
            batch_size=batch_size,
            seq_len=seq_len,
            use_dynamic_bsz=True,
            use_prefix_tree=False,
            max_token_len_per_gpu=max_token_len,
        )
        return prepare_micro_batches(td)

    def test_returns_micro_batches(self):
        micro_batches, _ = self._run()
        self.assertIsInstance(micro_batches, list)
        self.assertGreater(len(micro_batches), 0)

    def test_no_prefix_tree_subtree_when_disabled(self):
        """When use_prefix_tree=False, prefix_tree_subtree must NOT be attached."""
        micro_batches, _ = self._run()
        for mb in micro_batches:
            subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
            self.assertIsNone(subtree, "prefix_tree_subtree should not be present when use_prefix_tree=False")

    def test_batch_idx_list_covers_all_samples(self):
        batch_size = 4
        _, batch_idx_list = self._run(batch_size=batch_size)
        all_indices = set()
        for idx_group in batch_idx_list:
            all_indices.update(idx_group)
        self.assertEqual(all_indices, set(range(batch_size)))


# ---------------------------------------------------------------------------
# Tests: Path 3 — use_dynamic_bsz=False (fixed micro-batch size)
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesFixed(unittest.TestCase):
    """Path 3: use_dynamic_bsz=False, use_prefix_tree=False."""

    def _run(self, batch_size=4, seq_len=8, mbs_per_gpu=2):
        td = _make_data(
            batch_size=batch_size,
            seq_len=seq_len,
            use_dynamic_bsz=False,
            use_prefix_tree=False,
            micro_batch_size_per_gpu=mbs_per_gpu,
        )
        micro_batches, batch_idx_list = prepare_micro_batches(td)
        return micro_batches, batch_idx_list, td

    def test_returns_correct_number_of_micro_batches(self):
        micro_batches, _, _ = self._run(batch_size=4, mbs_per_gpu=2)
        self.assertEqual(len(micro_batches), 4 // 2)

    def test_batch_idx_list_is_none_for_fixed_path(self):
        _, batch_idx_list, _ = self._run()
        self.assertIsNone(batch_idx_list)

    def test_each_micro_batch_has_correct_size(self):
        micro_batches, _, _ = self._run(batch_size=4, mbs_per_gpu=2)
        for mb in micro_batches:
            self.assertEqual(len(mb), 2)

    def test_no_prefix_tree_subtree_when_disabled(self):
        micro_batches, _, _ = self._run()
        for mb in micro_batches:
            subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
            self.assertIsNone(subtree)


# ---------------------------------------------------------------------------
# Tests: Path 3 with prefix_tree=True
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesFixedWithPrefixTree(unittest.TestCase):
    """Fixed path (use_dynamic_bsz=False) + use_prefix_tree=True."""

    def _make_td(self, batch_size=4, seq_len=8, mbs_per_gpu=2):
        input_ids_list = [torch.arange(seq_len, dtype=torch.long) for _ in range(batch_size)]
        input_ids = torch.nested.as_nested_tensor(input_ids_list, layout=torch.jagged)
        td = TensorDict({"input_ids": input_ids}, batch_size=[batch_size])
        tu.assign_non_tensor(td, use_dynamic_bsz=False)
        tu.assign_non_tensor(td, use_prefix_tree=True)
        tu.assign_non_tensor(td, sp_size=1)
        tu.assign_non_tensor(td, force_group_size=1)
        tu.assign_non_tensor(td, micro_batch_size_per_gpu=mbs_per_gpu)
        td["micro_batch_size_per_gpu"] = torch.tensor(mbs_per_gpu)
        return td

    def test_fixed_path_with_prefix_tree_attaches_subtree(self):
        """Fixed path with use_prefix_tree=True must attach prefix_tree_subtree to all micro-batches."""
        td = self._make_td(batch_size=4, seq_len=8, mbs_per_gpu=2)
        micro_batches, _ = prepare_micro_batches(td)
        for i, mb in enumerate(micro_batches):
            subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
            self.assertIsNotNone(
                subtree,
                f"micro-batch {i} missing prefix_tree_subtree (fixed path, use_prefix_tree=True)",
            )


# ---------------------------------------------------------------------------
# Tests: Non-tensor segment arrays sliced correctly
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesNonTensorSegments(unittest.TestCase):
    """Non-tensor per-sample metadata must be correctly sliced per micro-batch."""

    def test_dynamic_path_segment_data_sliced_correctly(self):
        """Segment metadata (NonTensorStack) in micro-batches must match the global indices."""
        batch_size = 4
        seq_len = 8
        input_ids_list = [torch.arange(seq_len, dtype=torch.long) for _ in range(batch_size)]
        input_ids = torch.nested.as_nested_tensor(input_ids_list, layout=torch.jagged)
        td = TensorDict(
            {"input_ids": input_ids, "max_token_len_per_gpu": torch.tensor(512)},
            batch_size=[batch_size],
        )
        segment_data = [f"segment_{i}" for i in range(batch_size)]
        tu.assign_non_tensor(td, use_dynamic_bsz=True)
        tu.assign_non_tensor(td, use_prefix_tree=False)
        tu.assign_non_tensor(td, sp_size=1)
        tu.assign_non_tensor(td, force_group_size=1)
        tu.assign_non_tensor(td, prefix_segments_batch=segment_data)

        micro_batches, batch_idx_list = prepare_micro_batches(td)

        for idx_group, mb in zip(batch_idx_list, micro_batches, strict=False):
            segs = tu.get_non_tensor_data(mb, "prefix_segments_batch", default=None)
            if segs is not None:
                for local_i, global_i in enumerate(idx_group):
                    self.assertEqual(segs[local_i], segment_data[global_i])

    def test_fixed_path_microbatch_count_and_size(self):
        """Fixed path must produce (batch_size // mbs) micro-batches each of size mbs."""
        batch_size = 6
        mbs = 2
        seq_len = 8
        input_ids_list = [torch.arange(seq_len, dtype=torch.long) for _ in range(batch_size)]
        input_ids = torch.nested.as_nested_tensor(input_ids_list, layout=torch.jagged)
        td = TensorDict({"input_ids": input_ids}, batch_size=[batch_size])
        tu.assign_non_tensor(td, use_dynamic_bsz=False)
        tu.assign_non_tensor(td, use_prefix_tree=False)
        tu.assign_non_tensor(td, sp_size=1)
        tu.assign_non_tensor(td, force_group_size=1)
        tu.assign_non_tensor(td, micro_batch_size_per_gpu=mbs)
        td["micro_batch_size_per_gpu"] = torch.tensor(mbs)

        micro_batches, _ = prepare_micro_batches(td)
        self.assertEqual(len(micro_batches), batch_size // mbs)
        for mb in micro_batches:
            self.assertEqual(len(mb), mbs)


# ---------------------------------------------------------------------------
# Tests: Trie built exactly once (not per micro-batch)
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesTrieBuiltOnce(unittest.TestCase):
    """greedy_build_tries must be called exactly once per prepare_micro_batches call."""

    def test_greedy_build_tries_called_once(self):
        import verl.utils.prefix_tree.dynamic as _dyn

        td = _make_data(batch_size=4, seq_len=8, use_dynamic_bsz=True, use_prefix_tree=True, max_token_len_per_gpu=512)

        with patch.object(_dyn, "greedy_build_tries", wraps=_dyn.greedy_build_tries) as mock_build:
            prepare_micro_batches(td)
            # One call in prepare_micro_batches to build the global trie.
            # prepare_prefix_tree_micro_batches finds the pre-built trie in the data
            # and does NOT call greedy_build_tries again.
            self.assertEqual(mock_build.call_count, 1)


# ---------------------------------------------------------------------------
# Tests: Mock-isolated path dispatch
# ---------------------------------------------------------------------------


class TestPrepareMicroBatchesMockDependencies(unittest.TestCase):
    """Confirm path dispatch without running real trie/balancing logic."""

    def test_dynamic_prefix_path_calls_prepare_prefix_tree_micro_batches(self):
        """Path 1 must delegate to prepare_prefix_tree_micro_batches."""
        td = _make_data(batch_size=2, seq_len=4, use_dynamic_bsz=True, use_prefix_tree=True, max_token_len_per_gpu=256)
        mock_mb = MagicMock(spec=TensorDict)

        # The function is imported inside the dynamic+prefix branch of prepare_micro_batches
        # from verl.utils.prefix_tree.dynamic, so patch there.
        with patch(
            "verl.utils.prefix_tree.dynamic.prepare_prefix_tree_micro_batches",
            return_value=([mock_mb], [[0, 1]]),
        ) as mock_fn:
            micro_batches, batch_idx_list = prepare_micro_batches(td)
            mock_fn.assert_called_once()
            self.assertIs(micro_batches[0], mock_mb)
            self.assertEqual(batch_idx_list, [[0, 1]])

    def test_dynamic_only_path_calls_rearrange_micro_batches(self):
        """Path 2 must delegate to rearrange_micro_batches."""
        td = _make_data(batch_size=4, seq_len=8, use_dynamic_bsz=True, use_prefix_tree=False, max_token_len_per_gpu=512)
        mock_mb = MagicMock(spec=TensorDict)

        # rearrange_micro_batches is imported at module level in utils.py; patch it there.
        with patch.object(_engine_utils_mod, "rearrange_micro_batches", return_value=([mock_mb, mock_mb], [[0, 1], [2, 3]])) as mock_fn:
            prepare_micro_batches(td)
            mock_fn.assert_called_once()

    def test_fixed_path_calls_chunk_tensordict(self):
        """Path 3 must delegate to tu.chunk_tensordict."""
        batch_size = 4
        mbs = 2
        td = _make_data(batch_size=batch_size, seq_len=8, use_dynamic_bsz=False, use_prefix_tree=False, micro_batch_size_per_gpu=mbs)

        input_ids_list = [torch.arange(8, dtype=torch.long) for _ in range(mbs)]
        input_ids = torch.nested.as_nested_tensor(input_ids_list, layout=torch.jagged)
        fake_chunk = TensorDict({"input_ids": input_ids}, batch_size=[mbs])

        with patch.object(_engine_utils_mod.tu, "chunk_tensordict", return_value=[fake_chunk, fake_chunk]) as mock_chunk:
            micro_batches, batch_idx_list = prepare_micro_batches(td)
            mock_chunk.assert_called_once()
            self.assertIsNone(batch_idx_list)


if __name__ == "__main__":
    unittest.main()
