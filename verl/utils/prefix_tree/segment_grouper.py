# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""Segment-based grouping for prefix-tree: build trie from pre-computed segment
(hash, length) metadata, avoiding token-by-token detection."""

from typing import Hashable

import numpy as np


def create_segment_metadata(
    segments: list[list[tuple[Hashable, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Create (segment_hashes, segment_lengths) numpy arrays (object dtype, reorder-safe)."""
    segment_hashes = [[hash(h) & 0xFFFFFFFF for h, _ in segs] for segs in segments]
    segment_lengths = [[length for _, length in segs] for segs in segments]
    return (
        np.array(segment_hashes, dtype=object),
        np.array(segment_lengths, dtype=object),
    )


def group_by_segment_hash(
    segment_hashes: np.ndarray,
    segment_lengths: np.ndarray,
    level: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Group samples by segment hash at given level, returning hash → [(sample_idx, segment_length), ...]."""
    groups: dict[int, list[tuple[int, int]]] = {}
    for sample_idx, (sh, sl) in enumerate(zip(segment_hashes, segment_lengths, strict=False)):
        if level >= len(sh):
            continue
        hash_val = int(sh[level])
        length = int(sl[level])
        groups.setdefault(hash_val, []).append((sample_idx, length))
    return groups
