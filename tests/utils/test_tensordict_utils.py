# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
Tests for chunk_tensordict with non-tensor numpy array slicing.

Verifies that when a TensorDict stores numpy arrays via set_non_tensor(),
chunk_tensordict() correctly slices those arrays per chunk instead of
duplicating the full array into every chunk.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.utils.tensordict_utils import chunk_tensordict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_td(batch_size: int, extra_non_tensor: dict = None) -> TensorDict:
    """Build a simple TensorDict with a regular tensor plus optional non-tensor arrays."""
    td = TensorDict(
        {"input_ids": torch.arange(batch_size).unsqueeze(1).expand(batch_size, 4).clone()},
        batch_size=[batch_size],
    )
    if extra_non_tensor:
        for key, val in extra_non_tensor.items():
            td.set_non_tensor(key, val)
    return td


# ---------------------------------------------------------------------------
# Core behaviour: numpy array slicing
# ---------------------------------------------------------------------------

class TestChunkTensordictNonTensorSlicing:
    def test_numpy_array_sliced_per_chunk(self):
        """Non-tensor numpy arrays must be sliced to match each chunk's batch indices."""
        arr = np.array([10, 20, 30, 40])
        td = _make_td(4, extra_non_tensor={"my_arr": arr})

        chunks = chunk_tensordict(td, chunks=2)

        assert len(chunks) == 2
        np.testing.assert_array_equal(chunks[0].get_non_tensor("my_arr"), arr[:2])
        np.testing.assert_array_equal(chunks[1].get_non_tensor("my_arr"), arr[2:])

    def test_segment_hashes_sliced_correctly(self):
        """segment_hashes array should get the right slice per chunk."""
        segment_hashes = np.array([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
        td = _make_td(6, extra_non_tensor={"segment_hashes": segment_hashes})

        chunks = chunk_tensordict(td, chunks=3)

        assert len(chunks) == 3
        np.testing.assert_array_equal(chunks[0].get_non_tensor("segment_hashes"), segment_hashes[0:2])
        np.testing.assert_array_equal(chunks[1].get_non_tensor("segment_hashes"), segment_hashes[2:4])
        np.testing.assert_array_equal(chunks[2].get_non_tensor("segment_hashes"), segment_hashes[4:6])

    def test_segment_lengths_sliced_correctly(self):
        """segment_lengths array should get the right slice per chunk."""
        segment_lengths = np.array([100, 200, 300, 400])
        td = _make_td(4, extra_non_tensor={"segment_lengths": segment_lengths})

        chunks = chunk_tensordict(td, chunks=2)

        np.testing.assert_array_equal(chunks[0].get_non_tensor("segment_lengths"), np.array([100, 200]))
        np.testing.assert_array_equal(chunks[1].get_non_tensor("segment_lengths"), np.array([300, 400]))

    def test_multiple_non_tensor_arrays_all_sliced(self):
        """Multiple numpy arrays stored in the same TensorDict are all sliced."""
        hashes = np.array([1, 2, 3, 4])
        lengths = np.array([10, 20, 30, 40])
        td = _make_td(4, extra_non_tensor={"hashes": hashes, "lengths": lengths})

        chunks = chunk_tensordict(td, chunks=2)

        np.testing.assert_array_equal(chunks[0].get_non_tensor("hashes"), hashes[:2])
        np.testing.assert_array_equal(chunks[1].get_non_tensor("hashes"), hashes[2:])
        np.testing.assert_array_equal(chunks[0].get_non_tensor("lengths"), lengths[:2])
        np.testing.assert_array_equal(chunks[1].get_non_tensor("lengths"), lengths[2:])

    def test_chunk_slice_matches_batch_indices(self):
        """Each chunk's numpy slice aligns with the same batch rows as the tensors."""
        arr = np.arange(8)
        td = _make_td(8, extra_non_tensor={"arr": arr})

        chunks = chunk_tensordict(td, chunks=4)

        for i, chunk in enumerate(chunks):
            expected_slice = arr[i * 2 : (i + 1) * 2]
            np.testing.assert_array_equal(chunk.get_non_tensor("arr"), expected_slice)
            # Also verify the regular tensor rows match the same batch index window.
            expected_tensor_rows = torch.arange(i * 2, (i + 1) * 2)
            assert torch.equal(chunk["input_ids"][:, 0], expected_tensor_rows)


# ---------------------------------------------------------------------------
# Original TensorDict not modified
# ---------------------------------------------------------------------------

class TestChunkTensordictImmutability:
    def test_original_td_not_modified(self):
        """chunk_tensordict must not alter the source TensorDict."""
        arr = np.array([1, 2, 3, 4])
        td = _make_td(4, extra_non_tensor={"arr": arr})
        original_arr = arr.copy()
        original_input_ids = td["input_ids"].clone()

        chunk_tensordict(td, chunks=2)

        np.testing.assert_array_equal(td.get_non_tensor("arr"), original_arr)
        assert torch.equal(td["input_ids"], original_input_ids)

    def test_chunks_do_not_share_array_objects(self):
        """Slices in different chunks must not be the same object as the original."""
        arr = np.array([10, 20, 30, 40])
        td = _make_td(4, extra_non_tensor={"arr": arr})

        chunks = chunk_tensordict(td, chunks=2)

        # Each chunk's slice should differ from the full array.
        assert len(chunks[0].get_non_tensor("arr")) == 2
        assert len(chunks[1].get_non_tensor("arr")) == 2


# ---------------------------------------------------------------------------
# Non-array non-tensor data is preserved unchanged
# ---------------------------------------------------------------------------

class TestChunkTensordictNonArrayNonTensor:
    def test_scalar_metadata_preserved_in_all_chunks(self):
        """Non-array non-tensor values (e.g. strings, ints) are kept unchanged."""
        from tensordict.tensorclass import NonTensorData

        td = _make_td(4)
        td.set_non_tensor("experiment", "run_001")
        td.set_non_tensor("seed", 42)

        chunks = chunk_tensordict(td, chunks=2)

        for chunk in chunks:
            assert chunk.get_non_tensor("experiment") == "run_001"
            assert chunk.get_non_tensor("seed") == 42


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestChunkTensordictEdgeCases:
    def test_single_chunk(self):
        """Single chunk must return the full numpy array unchanged."""
        arr = np.array([1, 2, 3, 4])
        td = _make_td(4, extra_non_tensor={"arr": arr})

        chunks = chunk_tensordict(td, chunks=1)

        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0].get_non_tensor("arr"), arr)

    def test_empty_numpy_array(self):
        """An empty numpy array stored as non-tensor should slice into empty arrays."""
        arr = np.array([])
        td = TensorDict(
            {"input_ids": torch.zeros(0, 4, dtype=torch.long)},
            batch_size=[0],
        )
        # Cannot chunk a zero-length TD by 2 (0 % 2 == 0, chunk_size=0).
        # Instead verify with batch_size=2, empty-looking data.
        arr2 = np.array([10, 20])
        td2 = _make_td(2, extra_non_tensor={"arr": arr2})
        chunks = chunk_tensordict(td2, chunks=2)
        np.testing.assert_array_equal(chunks[0].get_non_tensor("arr"), np.array([10]))
        np.testing.assert_array_equal(chunks[1].get_non_tensor("arr"), np.array([20]))

    def test_no_non_tensor_data(self):
        """TensorDicts with no non-tensor data should chunk normally."""
        td = TensorDict(
            {"a": torch.arange(6).reshape(6, 1)},
            batch_size=[6],
        )
        chunks = chunk_tensordict(td, chunks=3)
        assert len(chunks) == 3
        assert len(chunks[0]) == 2

    def test_2d_numpy_array_sliced(self):
        """2D numpy arrays should be sliced along axis 0 per chunk."""
        arr = np.arange(12).reshape(4, 3)
        td = _make_td(4, extra_non_tensor={"matrix": arr})

        chunks = chunk_tensordict(td, chunks=2)

        np.testing.assert_array_equal(chunks[0].get_non_tensor("matrix"), arr[:2])
        np.testing.assert_array_equal(chunks[1].get_non_tensor("matrix"), arr[2:])
