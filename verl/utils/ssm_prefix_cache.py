"""SSM prefix cache with chunked rolling hash and LRU eviction.

Emulates vLLM's prefix KV-cache design for linear-attention (GDN) models.

vLLM KV-cache (transformer):
    - Split tokens into fixed-size chunks (blocks)
    - Cache K/V tensors per layer per chunk  →  O(layers × heads × chunk_size)
    - Chain hashes: h[i] = hash(chunk_i || h[i-1])  (Merkle-like)
    - On lookup: match as many leading chunks as possible; resume from the last hit

SSM prefix cache (linear attention, GDN):
    - SAME chunked rolling-hash scheme as vLLM
    - Cache ONE recurrent state per layer instead of K/V per token
      →  O(layers × state_size) per entry, independent of prefix length
    - Key difference: the recurrent state IS the complete sufficient statistic
      for all tokens processed so far.  There is no "partial chunk" state issue
      because the hash identifies the chunk BOUNDARY, not a mid-chunk position.
    - Linear attention uses ONLY the last chunk's hash as the cache key —
      the full chain of hashes leading up to it is used only during lookup
      to identify the deepest matching boundary.

Rolling hash scheme (matches vLLM block hashing):
    h[-1] = SEED (empty prefix sentinel)
    h[i]  = hash(tokens[i*K : (i+1)*K]  ||  h[i-1])

Reference:
    _hash_prefix():  verl/utils/prefix_tree_magi.py:314
    vLLM prefix caching:  vllm/core/block/prefix_caching_block.py

Usage:
    cache = SSMPrefixCache(capacity=512, chunk_size=16)

    # Before running sequence:
    hit_pos, snap = cache.lookup(token_ids)
    # hit_pos: how many tokens the restored snapshot covers (multiple of chunk_size)

    # After running sequence (save state at each new chunk boundary):
    cache.insert(token_ids, snapshots_per_boundary)
"""

from __future__ import annotations

import sys
import os
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Link to existing hash primitive from prefix_tree_magi.py
# ---------------------------------------------------------------------------

def _import_hash_fn():
    """Import _hash_prefix from verl if available; else use local fallback."""
    _verl_dir = os.path.join(os.path.dirname(__file__), "verl-prefix-tree")
    if _verl_dir not in sys.path:
        sys.path.insert(0, _verl_dir)
    try:
        from verl.utils.prefix_tree_magi import _hash_prefix
        return _hash_prefix
    except ImportError:
        pass

    # Fallback: same implementation
    def _hash_prefix(token_ids_flat: Tensor) -> int:
        raw = token_ids_flat.cpu().numpy().tobytes()
        try:
            import xxhash
            return xxhash.xxh128_intdigest(raw)
        except ImportError:
            import hashlib
            return int.from_bytes(hashlib.md5(raw).digest(), "little")

    return _hash_prefix


_hash_prefix = _import_hash_fn()

# Sentinel hash for the empty prefix (before any tokens)
_EMPTY_HASH: int = 0


# ---------------------------------------------------------------------------
# Rolling chunk hash
# ---------------------------------------------------------------------------

def chunk_rolling_hashes(
    token_ids: Tensor,   # 1-D, CPU
    chunk_size: int,
) -> list[int]:
    """Compute rolling chunk hashes for all complete chunks in token_ids.

    h[-1] = EMPTY_HASH
    h[i]  = _hash_prefix(cat([tokens[i*K:(i+1)*K], int64(h[i-1])])

    Returns list of length n_complete_chunks.
    Each h[i] uniquely identifies "the state of the sequence after (i+1)*K tokens",
    because the hash chain encodes all previous chunks.

    For linear attention, only h[-1] (the last complete chunk's hash) is used
    as the cache lookup key — the recurrent state at that boundary captures the
    full history.
    """
    ids = token_ids.flatten().cpu()
    n_complete = len(ids) // chunk_size
    hashes: list[int] = []
    prev = _EMPTY_HASH
    try:
        import xxhash as _xx
        _digest = lambda b: _xx.xxh128_intdigest(b)
    except ImportError:
        import hashlib as _hl
        _digest = lambda b: int.from_bytes(_hl.md5(b).digest(), "little")

    for i in range(n_complete):
        chunk = ids[i * chunk_size: (i + 1) * chunk_size]
        # Chain: hash(chunk_bytes || prev_hash_bytes)
        # prev is a 128-bit int; encode as 16 bytes little-endian
        chunk_bytes = chunk.to(torch.int32).numpy().tobytes()
        prev_bytes  = prev.to_bytes(16, "little")
        prev = _digest(chunk_bytes + prev_bytes)
        hashes.append(prev)
    return hashes


# ---------------------------------------------------------------------------
# SSMPrefixCache
# ---------------------------------------------------------------------------

class SSMPrefixCache:
    """LRU cache mapping chunk-boundary hash → SSM snapshot.

    Key  : int  — rolling hash at a chunk boundary (see chunk_rolling_hashes)
    Value: _NodeSnapshot (or any object with .clear() for cleanup)

    Capacity is measured in NUMBER OF ENTRIES (each entry = one chunk boundary).
    The total memory footprint is capacity × snapshot_size.

    Eviction policy: LRU — the least-recently-used entry is evicted when full.
    """

    def __init__(self, capacity: int = 512, chunk_size: int = 16):
        """
        Args:
            capacity:   Max number of cached chunk-boundary states.
            chunk_size: Tokens per chunk (must match the value used at insert time).
        """
        self.capacity = capacity
        self.chunk_size = chunk_size
        # OrderedDict: key → (boundary_token_pos, snapshot)
        # Ordered by insertion/access (LRU = first)
        self._store: OrderedDict[int, tuple[int, object]] = OrderedDict()

    # ---- Public API ----

    def lookup(self, token_ids: Tensor) -> tuple[int, Optional[object]]:
        """Find the deepest cached boundary that is a prefix of token_ids.

        Computes rolling hashes for all complete chunks in token_ids and walks
        backward from the deepest chunk to find the longest cached prefix.

        Returns:
            (hit_pos, snapshot) where hit_pos is the number of tokens covered
            by the cached state (always a multiple of chunk_size), or (0, None)
            if no match is found.

        For linear attention: because the recurrent state is the complete
        history, finding the last chunk's hash is sufficient — we don't need
        to verify the entire hash chain (it's embedded in the rolling hash).
        """
        hashes = chunk_rolling_hashes(token_ids, self.chunk_size)
        if not hashes:
            return 0, None

        # Walk deepest-first: try to find the longest matching prefix
        for i in range(len(hashes) - 1, -1, -1):
            h = hashes[i]
            if h in self._store:
                boundary_pos, snap = self._store[h]
                # LRU touch
                self._store.move_to_end(h)
                return boundary_pos, snap

        return 0, None

    def insert(
        self,
        token_ids: Tensor,
        snapshots: list,   # list[_NodeSnapshot], one per complete chunk boundary
    ) -> None:
        """Cache snapshots at each NEW chunk boundary in token_ids.

        snapshots[i] is the model state AFTER processing tokens[0 : (i+1)*chunk_size].
        Only boundaries not already in the cache are inserted.

        Evicts LRU entries as needed to stay within capacity.
        """
        hashes = chunk_rolling_hashes(token_ids, self.chunk_size)
        assert len(snapshots) == len(hashes), (
            f"snapshots length {len(snapshots)} != n_chunks {len(hashes)}"
        )

        for i, (h, snap) in enumerate(zip(hashes, snapshots)):
            if h in self._store:
                self._store.move_to_end(h)  # refresh LRU
                continue

            boundary_pos = (i + 1) * self.chunk_size
            self._evict_if_full()
            self._store[h] = (boundary_pos, snap)

    def evict(self, n: int = 1) -> None:
        """Explicitly evict the n least-recently-used entries."""
        for _ in range(min(n, len(self._store))):
            _, (_, snap) = self._store.popitem(last=False)
            if hasattr(snap, "clear"):
                snap.clear()

    def clear(self) -> None:
        """Evict all entries."""
        for _, (_, snap) in self._store.items():
            if hasattr(snap, "clear"):
                snap.clear()
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"SSMPrefixCache(capacity={self.capacity}, size={len(self)}, chunk_size={self.chunk_size})"

    # ---- Private ----

    def _evict_if_full(self) -> None:
        if len(self._store) >= self.capacity:
            _, (_, snap) = self._store.popitem(last=False)  # evict LRU
            if hasattr(snap, "clear"):
                snap.clear()


# ---------------------------------------------------------------------------
# Integration helper: run a sequence using the cache
# ---------------------------------------------------------------------------

def run_with_prefix_cache(
    model,
    token_ids: Tensor,          # (1, L) — full sequence
    cache: SSMPrefixCache,
    device: str = "cuda",
) -> tuple[Tensor, list]:
    """Run model forward on token_ids, reusing cached SSM states.

    1. Lookup the deepest cached chunk boundary matching token_ids.
    2. Restore that snapshot (if any) and run only the suffix.
    3. After the run, insert new chunk-boundary snapshots into the cache.

    Returns:
        (last_hidden_state (1,H), list of new _NodeSnapshot at each boundary)

    The caller is responsible for the cache lifecycle (clear at end of batch).
    """
    from ssm_branch_cache import (
        SSMBranchCache, _NodeSnapshot, apply_ssm_patch,
        _make_dyn_cache, _run_segment, _run_tokens_one_by_one, _text_model,
    )

    apply_ssm_patch()

    hit_pos, cached_snap = cache.lookup(token_ids.flatten())

    dyn = _make_dyn_cache(model)
    suffix = token_ids[:, hit_pos:]   # tokens not yet in cache

    if cached_snap is not None:
        cached_snap.restore_to(dyn)

    # Run prefix (chunk mode) then suffix one-by-one from cache hit point
    if hit_pos == 0:
        # No cache hit — run everything in chunk/prefill mode
        hidden = _run_segment(model, token_ids.to(device), dyn)
    else:
        # Suffix only (single-token decode from restored state)
        hidden = _run_tokens_one_by_one(model, suffix.to(device), dyn)

    # Collect snapshots at each new chunk boundary in the FULL sequence
    ids_full = token_ids.flatten().cpu()
    n_total_chunks = len(ids_full) // cache.chunk_size
    n_cached_chunks = hit_pos // cache.chunk_size
    new_snaps: list[_NodeSnapshot] = []

    # We need intermediate states — re-run up to each boundary from hit_pos
    # For efficiency, collect states during the forward pass by running chunk-by-chunk
    if n_total_chunks > n_cached_chunks:
        dyn2 = _make_dyn_cache(model)
        if cached_snap is not None:
            cached_snap.restore_to(dyn2)

        prev_snap = None
        for chunk_idx in range(n_cached_chunks, n_total_chunks):
            chunk_start = chunk_idx * cache.chunk_size
            chunk_end = chunk_start + cache.chunk_size
            chunk_tokens = ids_full[chunk_start:chunk_end].unsqueeze(0).to(device)

            if chunk_idx == n_cached_chunks and n_cached_chunks == 0:
                # First chunk: prefill mode
                _run_segment(model, chunk_tokens, dyn2)
            else:
                _run_tokens_one_by_one(model, chunk_tokens, dyn2)

            snap = _NodeSnapshot.from_dyn_cache(dyn2)
            new_snaps.append(snap)

        # Insert into cache (only the new chunks, skipping already-cached ones)
        all_snaps_for_new_boundaries = new_snaps
        # Only pass the suffix hashes and snaps to insert
        suffix_ids = ids_full[n_cached_chunks * cache.chunk_size:]
        cache.insert(suffix_ids, all_snaps_for_new_boundaries)

    return hidden[:, -1, :], new_snaps
