# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

RangeSpec = tuple[int, int]


@dataclass
class PrefixTreeParams:
    """Metadata for a flattened PrefixTree batch.

    ``multilevel`` should be set to ``True`` when the tree has more than one
    level of shared prefixes (built via ``build_multilevel_flex_spec``).  In
    that mode the strict single-level rectangle layout check is skipped.
    """

    prefix_range: RangeSpec
    prefix_segments: list[RangeSpec]
    leaf_ranges: list[RangeSpec]
    leaf_segments: list[RangeSpec]
    leaf_to_sample: list[int]
    sample_to_leaf_range: dict[int, RangeSpec]
    q_ranges: list[RangeSpec]
    k_ranges: list[RangeSpec]
    mask_types: list[str]
    total_seqlen_q: int
    total_seqlen_k: int
    flat_tokens: Optional[Tensor] = None
    flat_labels: Optional[Tensor] = None
    flat_loss_mask: Optional[Tensor] = None
    flat_position_ids: Optional[Tensor] = None
    multilevel: bool = False

    def __post_init__(self) -> None:
        if len(self.leaf_ranges) != len(self.leaf_segments):
            raise ValueError("leaf_ranges and leaf_segments must have the same length")
        if len(self.leaf_ranges) != len(self.leaf_to_sample):
            raise ValueError("leaf_ranges and leaf_to_sample must have the same length")
        if len(self.q_ranges) != len(self.k_ranges) or len(self.q_ranges) != len(self.mask_types):
            raise ValueError("q_ranges, k_ranges, and mask_types must have the same length")
        if set(self.leaf_to_sample) != set(self.sample_to_leaf_range):
            raise ValueError("sample_to_leaf_range must cover exactly the samples in leaf_to_sample")

        prefix_start, prefix_end = self.prefix_range
        if prefix_start != 0:
            raise ValueError("prefix_range must start at 0 in flattened PrefixTree layout")
        if prefix_end < prefix_start:
            raise ValueError("prefix_range must be non-decreasing")

        for leaf_range, leaf_segment in zip(self.leaf_ranges, self.leaf_segments):
            leaf_start, leaf_end = leaf_range
            if leaf_end < leaf_start:
                raise ValueError("leaf ranges must be non-decreasing")
            if leaf_segment != leaf_range:
                raise ValueError("leaf_segments must equal leaf_ranges")

        if self.total_seqlen_q != self.total_seqlen_k:
            raise ValueError("PrefixTree expects matching q/k sequence lengths")
        if self.leaf_ranges and self.leaf_ranges[-1][1] != self.total_seqlen_q:
            raise ValueError("last leaf range must end at total sequence length")
        if not self.leaf_ranges and self.prefix_range[1] != self.total_seqlen_q:
            raise ValueError("prefix-only PrefixTree must end at total sequence length")

        if not self.multilevel:
            # Single-level layout: validate exact rectangle structure.
            if self.prefix_segments != [self.prefix_range]:
                raise ValueError("root-shared PrefixTree expects prefix_segments to equal [prefix_range]")
            for leaf_start, _ in self.leaf_ranges:
                if leaf_start < prefix_end:
                    raise ValueError("leaf ranges must start at or after the shared prefix")
            expected_ranges = [self.prefix_range]
            expected_k_ranges = [self.prefix_range]
            expected_mask_types = ['causal']
            for leaf_range in self.leaf_ranges:
                expected_ranges.extend([leaf_range, leaf_range])
                expected_k_ranges.extend([self.prefix_range, leaf_range])
                expected_mask_types.extend(['full', 'causal'])
            if self.q_ranges != expected_ranges:
                raise ValueError("q_ranges do not match root-shared PrefixTree layout")
            if self.k_ranges != expected_k_ranges:
                raise ValueError("k_ranges do not match root-shared PrefixTree layout")
            if self.mask_types != expected_mask_types:
                raise ValueError("mask_types do not match root-shared PrefixTree layout")

        for sample_idx, leaf_range in zip(self.leaf_to_sample, self.leaf_ranges):
            if self.sample_to_leaf_range[sample_idx] != leaf_range:
                raise ValueError("sample_to_leaf_range does not match leaf_to_sample ordering")

        for name, tensor in {
            'flat_tokens': self.flat_tokens,
            'flat_labels': self.flat_labels,
            'flat_loss_mask': self.flat_loss_mask,
            'flat_position_ids': self.flat_position_ids,
        }.items():
            if tensor is not None and tensor.numel() != self.total_seqlen_q:
                raise ValueError(f"{name} must have total_seqlen_q elements")

        if self.flat_position_ids is not None and self.flat_position_ids.numel() > 0 and not self.multilevel:
            prefix_len = self.prefix_len
            if prefix_len > 0 and self.flat_position_ids[:prefix_len].tolist() != list(range(prefix_len)):
                raise ValueError("prefix positions must be contiguous starting at 0")
            for leaf_start, leaf_end in self.leaf_ranges:
                branch_len = leaf_end - leaf_start
                expected = list(range(prefix_len, prefix_len + branch_len))
                if self.flat_position_ids[leaf_start:leaf_end].tolist() != expected:
                    raise ValueError("leaf positions must continue after the shared prefix")

    @property
    def prefix_len(self) -> int:
        return self.prefix_range[1] - self.prefix_range[0]

    @property
    def branch_lengths(self) -> list[int]:
        return [end - start for start, end in self.leaf_ranges]

    @property
    def num_samples(self) -> int:
        return len(self.leaf_to_sample)

    def get_leaf_range(self, sample_idx: int) -> RangeSpec:
        return self.sample_to_leaf_range[sample_idx]
