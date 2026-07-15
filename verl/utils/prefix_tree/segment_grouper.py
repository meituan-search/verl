# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
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
"""Segment-based grouping for prefix-tree optimization.

Provides a generic interface for building prefix trees from pre-computed
segment metadata, avoiding expensive token-by-token detection when the
grouping structure is known at data creation time.

Segment Format:
    Each sample is described as a list of segments: [(hash, length), ...]
    - hash: Any hashable value (int, str, UUID) identifying the segment content
    - length: Number of tokens in the segment

Tree Building:
    Samples sharing the same hash for segment i share that prefix path.
    The tree is built by matching segment hashes level by level.

Examples:
    GRPO (single prompt, n rollouts):
        Sample 0: [("uuid-p0", 100), ("resp-hash-0", 50)]
        Sample 1: [("uuid-p0", 100), ("resp-hash-1", 45)]
        -> Share first segment (uuid-p0), diverge at second

    Multi-turn conversation:
        Sample 0: [("sys-hash", 20), ("turn1-uuid", 50), ("resp-hash", 30)]
        Sample 1: [("sys-hash", 20), ("turn1-uuid", 50), ("resp-hash", 35)]
        -> Share first two segments, diverge at third
"""

from typing import Any, Callable, Hashable, Optional

from torch import Tensor


# Default hash function for string UUIDs
def _default_uid_hash(uid: str) -> int:
    return hash(uid) & 0xFFFFFFFF


def create_segment_metadata(
    segments: list[list[tuple[Hashable, int]]],
    metadata: Optional[dict] = None,
) -> dict:
    """Create generic segment metadata for prefix-tree building.

    Args:
        segments: List of segments per sample. Each segment is (hash, length).
            Example: [[("uuid-0", 100), ("resp-0", 50)], [("uuid-0", 100), ("resp-1", 45)]]
        metadata: Optional additional metadata (e.g., {"rollout_n": 8, "type": "grpo"})

    Returns:
        Dict with segment metadata for non-tensor batch.
    """
    # Convert hashes to integers for consistent comparison
    int_segments = []
    for sample_segments in segments:
        int_segs = []
        for hash_val, length in sample_segments:
            if isinstance(hash_val, str):
                hash_val = _default_uid_hash(hash_val)
            elif not isinstance(hash_val, int):
                hash_val = hash(hash_val) & 0xFFFFFFFF
            int_segs.append((hash_val, length))
        int_segments.append(int_segs)

    result = {
        "segments": int_segments,
        "num_samples": len(segments),
    }
    if metadata:
        result["metadata"] = metadata
    return result


def create_grpo_segment_metadata(
    prompt_uids: list[str],
    prompt_lengths: list[int],
    rollout_n: int,
) -> dict:
    """Create segment metadata for GRPO prefix-tree grouping.

    In GRPO, each prompt generates `rollout_n` responses. Sequences from the
    same prompt share the prompt prefix.

    Args:
        prompt_uids: List of UUIDs for each sample's prompt. Same UUID = same prefix.
            Length = batch_size = num_prompts * rollout_n.
        prompt_lengths: Length of prompt (shared prefix) for each sample.
        rollout_n: Number of rollouts per prompt.

    Returns:
        Segment metadata dict compatible with build_tree_from_segments().
    """
    batch_size = len(prompt_uids)
    if batch_size % rollout_n != 0:
        raise ValueError(f"batch_size {batch_size} not divisible by rollout_n {rollout_n}")

    segments = []
    for uid, p_len in zip(prompt_uids, prompt_lengths):
        # GRPO: two segments - shared prompt + unique response
        segments.append([(uid, p_len)])

    return create_segment_metadata(
        segments,
        metadata={"type": "grpo", "rollout_n": rollout_n},
    )


def group_by_segment_hash(
    segments: list[list[tuple[int, int]]],
    level: int = 0,
) -> dict[int, list[tuple[int, int]]]:
    """Group samples by their segment hash at a given level.

    Args:
        segments: List of [(hash, length), ...] per sample.
        level: Which segment level to group by (0 = first segment).

    Returns:
        Dict mapping hash -> list of (sample_idx, segment_length).
    """
    groups: dict[int, list[tuple[int, int]]] = {}
    for sample_idx, sample_segments in enumerate(segments):
        if level >= len(sample_segments):
            continue
        hash_val, length = sample_segments[level]
        if hash_val not in groups:
            groups[hash_val] = []
        groups[hash_val].append((sample_idx, length))
    return groups


def validate_segment_metadata(
    metadata: dict,
    expected_type: Optional[str] = None,
    expected_rollout_n: Optional[int] = None,
) -> bool:
    """Validate segment metadata for tree building.

    Args:
        metadata: Dict from create_segment_metadata.
        expected_type: Expected metadata type (e.g., "grpo").
        expected_rollout_n: Expected rollout.n for GRPO.

    Returns:
        True if metadata is valid for fast-path tree building.
    """
    if metadata is None or "segments" not in metadata:
        return False

    if expected_type:
        meta = metadata.get("metadata", {})
        if meta.get("type") != expected_type:
            return False

    if expected_rollout_n:
        meta = metadata.get("metadata", {})
        if meta.get("rollout_n") != expected_rollout_n:
            return False

    return True
