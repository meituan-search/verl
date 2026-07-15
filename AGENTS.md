# Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## Profiling with nsys

verl has built-in nsys profiling support via `global_profiler` config. To profile actor updates:

```bash
# Profile step 2 (skip step 1 warmup), save to ~/nsys_profile/
python3 -m verl.trainer.main_ppo \
    +global_profiler.tool=nsys \
    "+global_profiler.steps=[2]" \
    ++global_profiler.save_path="$HOME/nsys_profile/magi_profile" \
    ++global_profiler.global_tool_config.nsys.discrete=true \
    '++global_profiler.global_tool_config.nsys.worker_nsight_options.trace=cuda,nvtx,cublas,ucx' \
    '++global_profiler.global_tool_config.nsys.worker_nsight_options.cuda-memory-usage=true' \
    '++global_profiler.global_tool_config.nsys.worker_nsight_options.cuda-graph-trace=graph' \
    '++global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi' \
    ++actor_rollout_ref.actor.profiler.enable=true \
    ++actor_rollout_ref.actor.profiler.all_ranks=true \
    ... # other config
```

Key requirements:
- `worker_nsight_options` must be set when using nsys with `profile_steps`
- `capture-range=cudaProfilerApi` is required for verl's profiler integration
- Use `discrete=true` for single-step profiling (avoids large files)
- Skip step 1 (warmup) to avoid initialization overhead in profile

See `prefix_script/run_grpo_longreason_magi_profile.sh` for a complete example.

---

## Prefix-Tree (MAGI) Architecture

### Overview
The prefix-tree attention system (MAGI) enables prefix-deduplicated training for LLMs. It packs multiple sequences with shared prefixes into a flat layout where shared tokens are processed once.

### Key Components

**Data Structures** (`verl/utils/prefix_tree/tree.py`):
- `TrieNode`: compressed trie node with `flat_idx`, `input_ids`, `ancestor`, `children`
- `PrefixTrie`: full batch view with flat `nodes` list indexed by `flat_idx`
- `PrefixSubTrie`: per-micro-batch view, serializable via `__getstate__`/`__setstate__`

**Layout Building** (`verl/utils/prefix_tree/utils.py`):
- `build_layout_from_tree_node()`: builds flat token layout and attention ranges
- `PrefixTreeParams`: holds `q_ranges`, `k_ranges`, `mask_types`, packed tensors
- Pre-packed labels avoid `torch.roll` cross-boundary issues

**MAGI Integration** (`verl/utils/prefix_tree/magi.py`):
- Dispatch happens **once per forward** (not per-layer)
- `get_position_ids(magi_key)` returns local token indices for CP rank
- Model receives pre-dispatched `local_input_ids` / `local_position_ids`
- `undispatch()` gathers local logits back to full layout for loss computation

**Attention Patch** (`verl/utils/prefix_tree/prefix_tree_patch_impl.py`):
- Patches `TEDotProductAttention.forward` to add MAGI/flex branches
- `magi_attn_forward()` calls `calc_attn()` from `magi_attention` package
- Fallback to FA3 if neither MAGI nor flex key is provided

### Configuration
```bash
# Enable prefix-tree with MAGI attention
actor_rollout_ref.model.use_prefix_tree=True
actor_rollout_ref.model.prefix_tree_attention=magi  # or "flex"
```

### Key Files
- `verl/utils/prefix_tree/magi.py`: main forward path
- `verl/utils/prefix_tree/tree.py`: data structures
- `verl/utils/prefix_tree/utils.py`: layout builder
- `verl/utils/prefix_tree/prefix_tree_patch_impl.py`: Megatron patches
- `prefix_script/run_grpo_*_magi*.sh`: example training scripts

### Reorder-safety: `leaf_idx` is the source of truth

The trie's `sequence_ids` and `leaves[]` are indexed by **original** sample position and go stale after `DataProto.reorder` (inside `_balance_batch`). `leaf_idx` (numpy array in `non_tensor_batch`) is fancy-indexed by reorder automatically and stays correct.

- `_build_global_trie` (ray_trainer.py) attaches both `meta_info["prefix_tree"]` (trie) and `non_tensor_batch["leaf_idx"]` (sample→leaf flat_idx).
- `prepare_prefix_tree_micro_batches` calls `mbs_groups_from_leaf_idx` / `trie_dfs_leaf_order_from_leaf_idx` (NOT the `*_from_trie` variants) when a trie is attached.
- `create_and_attach_subtrie_views` reads `leaf_idx` from each microbatch's `non_tensor_batch` to build the subtrie view.
- All `leaf_idx`-based paths raise `ValueError` on `-1` entries (sample without leaf), no silent skips.

**Sort order is tree→sort** (build trie, then `_balance_batch` reorders). The `leaf_idx`-driven grouping makes this safe.

### Metric aggregation: wrap floats in `Metric`

`engine_workers._postprocess_output` aggregates metrics via `allgather_dict_into_dict` (wraps each value as `[val]` per DP rank) + `chain.from_iterable` (flattens lists-of-lists). Raw floats wrapped as `[float]` crash `chain.from_iterable` because floats aren't iterable.

- Metrics added BEFORE allgather must be wrapped in `Metric(value, aggregation=...)` so `Metric.aggregate_dp` handles them.
- Metrics added AFTER allgather (`loss`, `grad_norm`, `lr`, `mfu`, `perf/*`) stay scalar and bypass the list branch.
- `prefix_tree/attn_fa3_fallback_ratio` uses `Metric(MEAN)`. Counter tracks only `fa3` and `total` (magi/flex distinction dropped; only FA3 fallback ratio matters).

### VLM-config models (Qwen3.5): `vision_model` gating

`vision_model=hasattr(hf_config, "vision_config")` at both call sites in `transformer_impl.py`. Do NOT add `and "pixel_values" in multi_modal_inputs`; that breaks the standard (non-prefix-tree) path for Qwen3.5 because `vision_model=True` is needed to trigger the VLM code path that handles M-RoPE's 3D `position_ids` internally.

The `has_vision_data` check lives only at the fused path's prefix-tree guard: `if use_prefix_tree and not (vision_model and has_vision_data):` (model_forward_fused.py).

### Tree-builder diagnostics

When `PrefixTreeParams.__post_init__` raises `last leaf range must end at total sequence length`, the cause is `_assign_offsets` (uses `len(node.input_ids)`) disagreeing with `_emit` (slices actual samples). The per-node trace log (added in `build_layout_from_tree_node` when `last_leaf_end != total_tokens`) prints each node's `flat_idx`, `iid_len`, `start`, `end`, `donated_in`, `donated_out`, `n_children`.

Invariant per node: `end - start == (1 if donated_in else 0) + iid_len - (1 if donated_out else 0)`. Any node violating this is the bug.

The OLP crash `no_padding_2_padding: token count mismatch` is a separate symptom: `sum(prompt_len + response_len) != actual_tokens_in_output`. Diagnostic log prints `prompt_lens`, `response_lens`, `sequence_lens` lists to identify which sample is short.

### CPU test for leaf_idx after reorder

`/tmp/claude/leaf_idx_tests/run_test.py`: tests `mbs_groups_from_leaf_idx`, `trie_dfs_leaf_order_from_leaf_idx`, `create_and_attach_subtrie_views` with a simulated `DataProto.reorder` permutation. Run with:
```bash
cd /tmp/claude/leaf_idx_tests && python3 run_test.py
```
Imports `dynamic.py` directly via `importlib.util` to skip the package `__init__` (which pulls `magi_attention`).

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
