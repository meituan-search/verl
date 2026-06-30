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
- `TrieNode` — compressed trie node with `flat_idx`, `input_ids`, `ancestor`, `children`
- `PrefixTrie` — full batch view with flat `nodes` list indexed by `flat_idx`
- `PrefixSubTrie` — per-micro-batch view, serializable via `__getstate__`/`__setstate__`

**Layout Building** (`verl/utils/prefix_tree/utils.py`):
- `build_layout_from_tree_node()` — builds flat token layout and attention ranges
- `PrefixTreeParams` — holds `q_ranges`, `k_ranges`, `mask_types`, packed tensors
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
- `verl/utils/prefix_tree/magi.py` — main forward path
- `verl/utils/prefix_tree/tree.py` — data structures
- `verl/utils/prefix_tree/utils.py` — layout builder
- `verl/utils/prefix_tree/prefix_tree_patch_impl.py` — Megatron patches
- `prefix_script/run_grpo_*_magi*.sh` — example training scripts

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
