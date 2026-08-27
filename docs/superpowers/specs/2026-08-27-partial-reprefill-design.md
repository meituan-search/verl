# Partial Re-prefill: Per-Token Case-Based π_b Construction

Last updated: 08/27/2026

- **Date**: 2026-08-27
- **Branch**: `dev/wsl_v1_staleness_dev0`
- **Status**: Design approved in conversation; pending user spec review
- **Builds on**: `2026-08-19-reprefill-decoupled-ppo-design.md`
- **Target deployment**: colocate async; new `partial_reprefill` trainer
  (sibling to `reprefill_decoupled`, parallel A/B path)

## 1. Problem

`reprefill_decoupled` (`verl/trainer/ppo/v1/trainer_reprefill_decoupled.py`)
today unconditionally re-prefills every consumed trajectory at sample time to
produce `new_rollout_log_probs` (π_b at W_sample). This is wasted work in two
common cases:

- **Fully-fresh trajectories** decoded at W_sample (e.g. just finished in the
  current step's generation phase, weight hasn't advanced). Re-prefilling
  tokens already at W_sample yields exactly `rollout_log_probs` — pure cost,
  zero value.
- **Partial-rollout trajectories** whose prefix was decoded at an older weight
  but whose suffix was decoded at W_resume after a partial_rollout resume
  (`verl/workers/rollout/llm_server.py:369`). The resume prefill already
  rebuilds the prefix KV cache at W_resume — re-prefilling the prefix again at
  sample time to extract `prompt_logprobs` is duplicate work. Worse, the
  suffix tokens are already at W_resume, so re-prefilling them is also wasted.

The optimization: distinguish three cases per trajectory and only pay the
reprefill cost where it produces new information.

## 2. Design overview

Implemented as a **new trainer** `partial_reprefill`, sibling to
`reprefill_decoupled` under `PPOTrainerColocateAsync`. The existing
`reprefill_decoupled` is left untouched as the "always full reprefill"
reference path for A/B comparison.

- File: `verl/trainer/ppo/v1/trainer_partial_reprefill.py`
- Class: `PPOTrainerPartialReprefill(PPOTrainerColocateAsync)`
- Registration: `@register_trainer("partial_reprefill")`
- Config namespace: `trainer.v1.partial_reprefill.*`

The new trainer duplicates (not inherits) the relevant methods from
`reprefill_decoupled` — `on_train_begin`, `on_train_end`, `_on_new_finished`,
`on_sampled`, `_reprefill_all`, `_cancel_pending_prefills`,
`_compute_old_log_prob`, `_compute_advantage`,
`_debug_log_prob_extra_fields` — and overrides the case-decision logic in
`_compute_new_rollout_log_prob` / `_compute_new_rollout_log_prob_pipelined`.

Three cases, each handled at a different point in the pipeline:

| Case | Trigger | Handler | `new_rollout_log_probs` source |
|---|---|---|---|
| (3) fully-fresh | sample-time, all tokens at current W | `partial_reprefill` trainer | copy of `rollout_log_probs` (no reprefill) |
| (1) partial piggyback | partial_rollout resume time | `FullyAsyncLLMServerClient` | prefix: resume-prefill `prompt_logprobs` @ W_resume; suffix: copy of `rollout_log_probs` |
| (2) fully-stale | sample-time, no piggyback marker | `partial_reprefill` trainer | full reprefill @ W_sample |

Case decision is **emergent from data**, not centrally decided:

- Case (1) is detected by a per-trajectory piggyback marker (`new_rollout_log_probs` field already populated in TQ by the client).
- Case (3) vs (2) is decided by comparing the trainer's current parameter
  version against the **last token's version** in `token_versions` (per-trajectory
  scalar suffices because all tokens in a non-partial trajectory share one
  version; partial trajectories are intercepted by case 1 before reaching this
  check).

## 3. TQ schema

Per-trajectory fields in TQ:

| Field | Status | Semantics |
|---|---|---|
| `rollout_log_probs` | unchanged | decode-time logprob; partial trajectories naturally mixed (prefix @ W_prefix + suffix @ W_resume) |
| `new_rollout_log_probs` | **semantic change** | "W_resume-consistent π_b": partial → prefix re-prefill + suffix copy; fully-fresh → copy of `rollout_log_probs`; fully-stale → full reprefill. May be populated by client (case 1) or trainer (cases 2, 3). |
| `token_versions` | **new** | per-token int32 jagged, absolute `global_steps` at decode time per token. Partial trajectory naturally `[k,...,k, m,...,m]`. |
| `resume_version` tag | semantic change | the W at which `new_rollout_log_probs` is consistent. Case 1 = W_resume; case 2 = W_sample; case 3 = W_sample. |
| `piggyback_marker` tag | **new** | bool; true when client has populated `new_rollout_log_probs` (case 1). Trainer checks this first. |

### Downstream consumers

- `_compute_old_log_prob` (`trainer_reprefill_decoupled.py:335`): unchanged
  behavior — `old_log_probs = new_rollout_log_probs` rename. The mixed-source
  nature of `new_rollout_log_probs` is transparent: downstream sees a single
  per-trajectory π_b tensor.
- `compute_rollout_correction_and_add_to_batch`
  (`verl/trainer/ppo/rollout_corr_helper.py:1137`): consumes `old_log_probs`,
  `rollout_log_probs`, `new_rollout_log_probs` as opaque padded tensors. Shape
  contract unchanged. No modification needed.
- `calculate_debug_metrics` (`verl/utils/debug/metrics.py:63`): opaque tensor
  pairwise diffs. No modification needed.
- `compute_and_emit_staleness_metrics` (`reprefill_utils.py:84`): **unchanged**
  per user direction. Existing per-trajectory metrics (rollout/new_rollout/old
  pairwise) keep their per-trajectory semantics. New **token-level** metrics
  are added alongside (see §6).

## 4. Component changes

### 4.1 `FullyAsyncLLMServerClient.generate` (`llm_server.py:281-424`)

1. **Resume path requests `prompt_logprobs`**: in the partial_rollout resume
   branch (`llm_server.py:369`), the resume sampling params must include
   `prompt_logprobs=0` (top-1 logprob per prompt token) so SGLang emits
   prompt_logprobs for the prefix during the resume prefill. Currently the
   agent-loop sampling params (`agent_loop_tq.py:64-70`) only set `logprobs`
   (response-side); resume needs the prompt-side flag.
   - The existing `reprefill_trajectories` helper
     (`reprefill_utils.py:69-81`) already passes `prompt_logprobs=0,
     max_new_tokens=0` to extract prefix logprobs — the resume path adopts
     the same flag but with `max_new_tokens=N` to continue generation.

2. **`llm_server.py:394-395` overwrite behavior is correct for our "latest
   only" semantic** (per user direction §1 of confirmation): the current
   code overwrites `output.extra_fields["prompt_logprobs"]` across segments,
   so only the **last resume segment's** prompt_logprobs survive. This is
   exactly what we want — the last resume's prompt is `prompt_ids +
   cumulative_prefix`, so its prompt_logprobs cover the entire prefix at
   W_resume. Earlier segments' prompt_logprobs (at older W) are correctly
   discarded. **No code change needed at :394-395** beyond ensuring
   `prompt_logprobs` is requested (point #1); the existing overwrite is the
   desired behavior. The slice for `new_rollout_log_probs` prefix portion
   must skip `prompt_ids` (use the existing `slice_response_logprobs` helper
   with `prompt_len=len(prompt_ids), response_len=prefix_len`).

3. **Multi-interruption semantic: keep only the latest prefix reprefill**.
   If the resume itself is aborted again, the next resume re-prefills the
   entire prefix (previous prefix + previous suffix) at the new W. The
   earlier prefix's reprefill (at the older W) is discarded. Rationale:
   the earlier reprefill is at an older W; the latest resume's reprefill at
   the newer W supersedes it for `new_rollout_log_probs` purposes. The
   `token_versions` array, however, **accumulates** across all segments
   (each token retains its actual decode-time version).

4. **Build `token_versions` array**: each segment fills its tokens with the
   `global_steps` value from `output.extra_fields["global_steps"]` (set
   server-side at `async_sglang_server.py:680`, populated via
   `set_global_steps` at :705-707). Concatenate per segment across the
   partial_rollout loop. Single-segment (non-partial) trajectories get a
   uniform array.

5. **Build `new_rollout_log_probs` for partial trajectories only**:
   - Prefix portion: the prompt_logprobs emitted during the **latest** resume
     prefill (these are the prefix tokens' logprobs at W_resume).
   - Suffix portion: copy of `rollout_log_probs` for the suffix tokens
     (suffix was decoded at W_resume, so decode logprob == W_resume logprob;
     copy is equivalent to a fresh re-prefill at W_resume, but free).
   - Concatenate and write to TQ as `new_rollout_log_probs`. Set
     `piggyback_marker=true`, `resume_version=W_resume`.
   - Non-partial trajectories: **do not** write `new_rollout_log_probs` in
     the client; leave the field absent. Trainer decides case 2 or 3.

6. **Piggyback failure path**: if the resume's `prompt_logprobs` is missing or
   malformed (SGLang error, timeout, missing field), the client must NOT set
   `piggyback_marker`. Trainer will then fall through to case 2 (full
   reprefill at sample time). Log a warning.

### 4.2 Trainer `_compute_new_rollout_log_prob` (new file `trainer_partial_reprefill.py`)

The case-decision dispatcher. Per-trajectory loop:

```
for each trajectory in batch:
    if piggyback_marker:
        skip  # case 1: client already populated new_rollout_log_probs
    elif token_versions[-1] == current_parameter_version:
        copy rollout_log_probs → new_rollout_log_probs  # case 3
        resume_version = current_parameter_version
    else:
        full reprefill  # case 2: existing _reprefill_all + slice logic
        resume_version = current_parameter_version - 1  # W_{k-1}
```

Where `current_parameter_version` is the trainer's notion of the rollout
engine's current weight. **Value**: `self.global_steps - 1` (the same
expression used by the existing `_compute_new_rollout_log_prob` for
`resume_version` at `trainer_reprefill_decoupled.py:196, :265`). This works
because in colocate async the rollout engine holds W_{k-1} throughout the
window between `on_step_end` (sync weights to W_k) and the next
`on_step_end` — at sample time of step k, `global_steps` is k, so
`global_steps - 1 = k-1` matches the rollout engine's current W.

`token_versions[-1]` (last token's version) is the trajectory's decode
version. For non-partial trajectories (the only ones reaching the case 3
check, since partials are intercepted by `piggyback_marker`), all tokens
share one version and `token_versions[-1]` is representative. Per user
direction, recording a per-trajectory "last version" scalar at decode time
is also acceptable — either approach works.

Why `token_versions[-1]` and not the full array for case 3: by the time we
reach the case 3 check, partial trajectories have already been intercepted by
`piggyback_marker`. The remaining trajectories are single-segment, so all
tokens share one version and `token_versions[-1]` is representative. Per
user direction, a per-trajectory "last version" scalar recorded at decode
time is also acceptable — either approach works.

### 4.3 P2 pipelined path (`_compute_new_rollout_log_prob_pipelined`, new file)

`_on_new_finished` (duplicated from `reprefill_decoupled.py:116` into the new
trainer) currently pre-dispatches reprefill for every newly-finished
trajectory. Update:

- Skip pre-dispatch if `piggyback_marker` is set (case 1: client already
  did the work).
- Skip pre-dispatch if `token_versions[-1] == current_parameter_version`
  (case 3: no reprefill needed).
- Pre-dispatch only for case 2 trajectories.

The sample-time `_compute_new_rollout_log_prob_pipelined` consumes
pre-dispatched futures only for case 2; case 1 and 3 take the fast path
(no future lookup).

### 4.4 `compute_and_emit_staleness_metrics` (`reprefill_utils.py:84-130`)

**Unchanged** per user direction. Existing per-trajectory metrics:

- `offpolicy/staleness/*` (rollout vs new_rollout): for case 1 this measures
  prefix staleness only (suffix diff is zero because new_rollout suffix =
  rollout suffix = decode logprob at W_resume). For case 2 it's full
  staleness. For case 3 it's zero.
- `offpolicy/mismatch/*` (new_rollout vs old = new_rollout): always zero
  under `partial_reprefill` (old = new_rollout by rename, same as
  `reprefill_decoupled`). Unchanged.
- `offpolicy/combined/*` (rollout vs old): same as staleness under reprefill
  semantics.

### 4.5 New token-level metrics (new function, alongside existing)

Add `compute_and_emit_token_staleness_metrics(batch, metrics, global_steps)`
in `reprefill_utils.py`. Emits:

- `offpolicy_token/staleness_mean`: per-token mean of
  `|log(π_rollout) - log(π_new_rollout)|`, averaged across all tokens in
  the batch. Uses `token_versions` to gate: only tokens with
  `token_versions[i] < global_steps` contribute (others have zero staleness
  by construction).
- `offpolicy_token/staleness_by_version_gap`: histogram or bucketed mean
  by `global_steps - token_versions[i]` (0, 1, 2-3, 4+).
- `offpolicy_token/fresh_token_ratio`: fraction of tokens with
  `token_versions[i] == global_steps` (case 3 + case 1 suffix tokens).
- `offpolicy_token/stale_token_ratio`: complement.
- `offpolicy_token/case_distribution`: {case_1, case_2, case_3} trajectory
  counts in the batch.

Existing per-trajectory metrics remain under the `offpolicy/` prefix; new
token-level metrics live under `offpolicy_token/`. No regression in existing
dashboard series.

## 5. Data flow walkthrough

### Case 1: partial piggyback

```
W_k: generate prefix → abort (on_sample_end) → 1s sleep, weight syncs to W_{k+1}
  → resume at W_{k+1}:
      sampling_params = {logprobs=..., prompt_logprobs=0, max_new_tokens=N}
      → SGLang resume prefill:
          - rebuilds KV cache for prefix (existing behavior, no extra cost)
          - emits prompt_logprobs[prefix] @ W_{k+1}  (NEW: piggyback target)
      → continue decode suffix @ W_{k+1}:
          - emits rollout_log_probs[suffix] @ W_{k+1}  (existing behavior)
  → client builds:
      new_rollout_log_probs = prompt_logprobs[prefix] ∘ rollout_log_probs[suffix]
      token_versions = [k]*prefix_len + [k+1]*suffix_len
      piggyback_marker = true, resume_version = k+1
  → TQ put
  → trainer sample: piggyback_marker=true → case 1, skip
  → _compute_old_log_prob: old_log_probs = new_rollout_log_probs (rename)
  → PPO loss uses mixed-source π_b @ W_{k+1}
```

### Case 2: fully-stale (no piggyback)

```
W_old: trajectory fully decoded @ W_old, no interruption
  → client puts rollout_log_probs @ W_old, token_versions = [old]*N
     (no new_rollout_log_probs, no piggyback_marker)
  → trainer sample (now at W_k):
      piggyback_marker absent, token_versions[-1] = old ≠ k → case 2
      full reprefill @ W_k → new_rollout_log_probs
  → _compute_old_log_prob: old_log_probs = new_rollout_log_probs
```

### Case 3: fully-fresh (no piggyback)

```
W_k: trajectory fully decoded @ W_k during current step's generation phase
  → client puts rollout_log_probs @ W_k, token_versions = [k]*N
  → trainer sample (still at W_k, before next on_step_end):
      piggyback_marker absent, token_versions[-1] = k == k → case 3
      copy rollout_log_probs → new_rollout_log_probs
  → _compute_old_log_prob: old_log_probs = new_rollout_log_probs
```

## 6. Edge cases and error handling

- **partial_rollout multiple interruptions**: each abort-resume cycle
  re-prefills the cumulative prefix at the new W. Earlier cycles' reprefill
  results are discarded (superseded). `token_versions` accumulates per
  segment (each token keeps its actual decode version). The final
  `new_rollout_log_probs` reflects only the latest prefix reprefill + the
  final suffix. **Confirmed by user.**

- **piggyback failure**: client sets `piggyback_marker=false` (or omits it)
  on any SGLang error / missing `prompt_logprobs` field during resume.
  Trainer falls through to case 2. Log warning at client side, emit
  `reprefill_decoupled/piggyback_failure` counter.

- **W advances between resume and sample (case 1)**: `new_rollout_log_probs`
  is at W_resume, sample-time W is W_sample > W_resume. This is standard
  staleness, handled by TIS/MIS/RS correction (`compute_rollout_correction_and_add_to_batch`).
  No re-reprefill. Matches existing `reprefill_decoupled` semantics for
  carried-over trajectories — the new trainer preserves this behavior.

- **Case 3 false-positive**: trainer checks `token_versions[-1] ==
  current_parameter_version`. For partial trajectories, suffix version is
  W_resume; if W_resume == W_sample, suffix is fresh — but prefix is
  W_prefix < W_sample, so the trajectory is not "all fresh". This is
  correctly intercepted by `piggyback_marker` (case 1) before reaching the
  case 3 check. Partial trajectories never enter case 3.

- **Stale `current_parameter_version`**: the trainer's
  `_rollout_engine_version` field must be updated synchronously with
  `on_step_end`'s weight sync. If it lags, case 3 is over-applied (trajectories
  appear fresh when they're not). Mitigation: update in the same critical
  section as the weight sync call; assert on step transition.

- **Mixed-source `new_rollout_log_probs` shape contract**: case 1 produces a
  per-trajectory jagged tensor with prefix + suffix concatenated, total
  length = response_len. Same shape as case 2/3. Downstream
  `to_padded_tensor()` handles it opaquely. No contract change.

## 7. Testing strategy

### Unit tests

- `slice_response_logprobs` + multi-segment merge: mock SGLang returning
  per-segment `prompt_logprobs` and `logprobs`; verify
  `new_rollout_log_probs` is correctly assembled (prefix from
  prompt_logprobs, suffix from rollout_log_probs).
- `token_versions` assembly: verify multi-segment concatenation, single
  segment uniformity.
- Case decision logic: given `piggyback_marker`, `token_versions[-1]`,
  `current_parameter_version`, verify correct case selection.

### Integration tests (staleness_sweep trainer)

The staleness_sweep trainer (`trainer_staleness_sweep.py`) provides a
controlled environment for partial_rollout: manually abort a generation
mid-flight, advance W, resume. Verify:

- Case 1 trajectory's `new_rollout_log_probs` is non-stale on prefix
  (matches a fresh full reprefill at W_resume within tolerance) and
  identical to `rollout_log_probs` on suffix.
- Case 2 trajectory's `new_rollout_log_probs` matches a full reprefill at
  W_sample.
- Case 3 trajectory's `new_rollout_log_probs` matches `rollout_log_probs`
  exactly (no reprefill was performed — verify via timing / counter).

### Metrics invariants

- Case 1 trajectory: `offpolicy/staleness` (per-trajectory, mean over
  tokens) is non-zero on prefix tokens, zero on suffix tokens. New
  `offpolicy_token/staleness_mean` reflects this correctly.
- Case 2: all tokens non-zero.
- Case 3: all tokens zero.
- `offpolicy_token/case_distribution` sums to batch size.

### A/B test (compare mode)

Under `compare_trainer_old_log_prob=true` (flag duplicated from
`reprefill_decoupled` into the new trainer's config namespace
`trainer.v1.partial_reprefill.*`):

- Case 2 trajectory: `trainer_old_log_probs` (trainer forward @ W_train)
  should approximately match `new_rollout_log_probs` (reprefill @ W_sample),
  since W_train ≈ W_sample. Small T/R mismatch expected.
- Case 1 trajectory: same expectation, but only on prefix tokens (suffix
  has no trainer forward component in `new_rollout_log_probs` — it's a
  copy of `rollout_log_probs`).
- Case 3: `trainer_old_log_probs` should match `rollout_log_probs` (both
  at W_sample).

### Performance regression

- `new_rollout_log_prob` timer should drop proportionally to
  `offpolicy_token/case_3_ratio` (fully-fresh trajectories skip reprefill
  entirely).
- P2 pipelined path: `_pending_prefill` dict size should drop (case 1 + 3
  trajectories no longer pre-dispatched).

## 8. Rollout / migration

- Implemented as a new trainer registered as `partial_reprefill`. Users
  opt in via `trainer.v1.trainer_mode=partial_reprefill`. Existing
  `reprefill_decoupled` trainer is untouched — no migration risk to its
  users.
- Feature sub-flags under `trainer.v1.partial_reprefill.*`:
  - `enable_prefill_pipeline` (default: false) — P2 pipelined pre-dispatch,
    mirrors the `reprefill_decoupled` flag.
  - `enable_case_skip` (default: true) — case 3 fast path.
  - `enable_piggyback` (default: true) — case 1 consumption of
    client-side piggyback. When false, ignores piggyback marker and
    always does case 2 full reprefill (for A/B isolation of the piggyback's
    effect).
  - `compare_trainer_old_log_prob` (default: false) — A/B timing mode,
    mirrors `reprefill_decoupled`.
- Phased rollout:
  1. Phase A: case 3 (skip reprefill for fully-fresh) only. Lowest risk,
     immediate measurable win. Piggyback disabled.
  2. Phase B: case 1 (partial_rollout piggyback). Requires client-side
     changes (`llm_server.py`), higher risk.
  3. Phase C: case 2 unchanged (existing full reprefill logic), but case
     decision dispatcher is active for all three cases.

## 9. Open questions for implementation plan

- **`current_parameter_version` plumbing**: where exactly in
  `trainer_colocate_async.py` / `trainer_base.py` is the weight sync
  observable? Need to identify the hook point for
  `_rollout_engine_version` update. (Likely `on_step_end` after
  `update_weights` returns.)
- **SGLang `prompt_logprobs` during prefill-with-generation**: verify that
  setting `prompt_logprobs=0` in a resume call (where `max_new_tokens > 0`)
  actually emits prefix logprobs. The standalone `reprefill_trajectories`
  path uses `max_new_tokens=0`; the resume path is different. May need a
  SGLang-side check or a separate prefill call after resume completes.
- **`agent_loop_tq.py` sampling params override**: the resume path needs
  different sampling params than the initial generate call. Where is the
  cleanest injection point?
