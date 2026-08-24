# Re-prefill Decoupled PPO (`reprefill_decoupled`) Design Spec

Last updated: 08/20/2026

- **Date**: 2026-08-19
- **Branch**: `dev/wsl_v1_staleness_dev0`
- **Status**: Design approved in conversation; pending user spec review
- **Target deployment**: colocate async first (P1/P2), separate async later (P3)

## 1. Problem

In async training, samples are consumed at trainer weight W_{k-1} but were decoded
at rollout weight W_0 (staleness = k-1 steps). The PPO ratio π_θ/π_rollout grows
with staleness, and past a threshold training collapses (entropy collapse / KL
explosion).

The existing **Decoupled PPO** mode (`verl/trainer/ppo/v1/trainer_base.py:1514`
`_compute_old_log_prob`, config `algorithm.rollout_correction`) fixes this by
recomputing π_b (`old_log_probs`) with a **trainer-engine forward pass** at a
stable version: in `trainer_separate_async.py:103` this requires
`save_model_to_cpu` / `restore_model_from_cpu` to keep π_old version-stable
across a `parameter_sync_step` cycle. This costs a full trainer-side inference
pass plus weight swap serialization.

## 2. Idea

The re-prefill machinery validated by the `staleness_sweep` trainer
(`verl/trainer/ppo/v1/trainer_staleness_sweep.py`) shows we can obtain
log probs of response tokens under the **rollout engine's** current weight
W_{k-1} via a re-prefill (`max_new_tokens=0, prompt_logprobs=0`). In colocate
async, the rollout engine holds W_{k-1} throughout the window between
`on_step_end` (update_weights) and `on_sample_end` (abort+sleep), so a re-prefill
issued during that window is **version-aligned with the decoupled π_old by
construction** — no CPU weight snapshot/restore needed.

Replace the trainer-side π_b computation with rollout-side re-prefill:

- `old_log_probs := new_rollout_log_probs` (re-prefill at W_{k-1}).
- Downstream TIS/MIS/RS correction (`compute_rollout_correction_and_add_to_batch`,
  `verl/trainer/ppo/rollout_corr_helper.py:1145`) and actor loss are unchanged —
  they consume `old_log_probs` as π_b regardless of where it came from.
- Residual risk vs trainer-side π_b is the T/R mismatch (kernel/numeric
  differences between engines), already measurable via the `mismatch/*` metrics
  from `compute_offpolicy_metrics`.

### Roadmap (A → B → C)

This spec covers **方案 A** (all consumed samples re-prefilled). Pre-approved
follow-ups, not in scope here:

- **B (trajectory-level)**: only re-prefill trajectories containing stale tokens
  (selected via `resume_version` / `min_global_steps` tags); bypass others
  (π_b = `rollout_log_probs`). Legal per-sample IS mixture, but different
  bias/variance — needs observation.
- **C (token-level, near-zero cost)**: `verl/workers/rollout/llm_server.py:388`
  resume loop already re-prefills the prefix under (possibly newer) weights on
  partial-rollout resume; adding `prompt_logprobs` to that request makes stale
  token log probs a free by-product. Apply correction only to stale tokens.

## 3. New trainer: `reprefill_decoupled`

`PPOTrainerReprefillDecoupled(PPOTrainerColocateAsync)`, registered as
`reprefill_decoupled`, enabled via
`trainer.v1.trainer_mode=reprefill_decoupled`.

### 3.1 Shared re-prefill helpers (extract from staleness_sweep)

Move to a shared module (e.g. `verl/trainer/ppo/v1/utils.py` or a new
`verl/trainer/ppo/v1/reprefill_utils.py`):

- `_build_reprefill_inputs(keys, partition_id)` — fetch prompts/responses from
  TQ, build token lists + real lens.
- `_slice_response_logprobs(prompt_logprobs_ls, prompt_len, response_len)` —
  sglang off-by-one slicing of response-token log probs.
- `_reprefill_trajectories_async(prompt_ids_list, sampling_params_list)` —
  async gather of `client.generate` with `max_new_tokens=0, prompt_logprobs=0`.
- Nested-jagged-tensor storage format for the result (must stay
  `torch.nested.as_nested_tensor(..., layout=torch.jagged)` so
  `KVBatchMeta.to_padded_tensor()` handles it — see
  trainer_staleness_sweep.py:168 comment).

`staleness_sweep` trainer refactored to consume the same helpers (behavior
unchanged).

### 3.2 Trainer behavior

**`on_sampled(batch, metrics)`** (override):

1. Re-prefill all consumed trajectories (P1: synchronous; P2: pipelined, see §5).
2. Write `new_rollout_log_probs` (nested jagged) to TQ.
3. Tag each key with `resume_version = global_steps - 1`.
4. Emit diagnostics via `compute_offpolicy_metrics` (staleness/mismatch/combined)
   — keeps observability of π_b quality.

**`_compute_old_log_prob(batch, metrics)`** (override):

- Fetch `new_rollout_log_probs` from TQ, write as `old_log_probs`.
- Skip the trainer forward pass entirely.
- When `algorithm.rollout_correction.bypass_mode=True`, delegate to the parent
  (bypass stays available for A/B experiments).

**Everything else inherited from `PPOTrainerColocateAsync`** — including
`on_sample_end` abort+sleep and `on_step_end` update_weights+resume.

### 3.3 `parameter_sync_step > 1` semantics

Within a `parameter_sync_step` cycle, all mini-batches must use the same π_b
version. Rollout engine weight is constant across the cycle (updates happen in
`on_step_end`), so re-prefill issued for any mini-batch in the cycle yields the
same version — the stability that separate_async achieves via CPU save/restore
is obtained for free. No extra mechanism needed; document this invariant.

## 4. P1 — correctness milestone

Synchronous re-prefill inside `on_sampled`: issue all requests, `await` all,
write to TQ. Simple and correct; the `on_sample_end` abort cannot race because
`on_sampled` returns only after all re-prefills finish.

Acceptance (user-defined): **stable, no collapse** — entropy does not collapse,
KL does not explode, reward grows normally. Three-arm comparison:

1. bypass mode, high staleness — reproduces the collapse (negative control);
2. `reprefill_decoupled` — no collapse (the claim);
3. existing decoupled trainer-π_b — no collapse (upper-bound reference).

## 5. P2 — pipelined re-prefill (incremental pre-dispatch)

Motivation: during `replay_buffer.sample()`'s poll loop
(`replay_buffer.py:432`), samples finish one by one; re-prefilling them as they
finish overlaps the re-prefill cost with the remaining generation time, so
`on_sampled` only waits for tail stragglers.

### Version correctness of pre-dispatch

- Newly finished samples pre-dispatched **during the current window** are served
  at W_{k-1} → exactly the π_b version step k needs. Valid.
- Samples finished in an **earlier** window get re-prefilled at W_{k-2} → wrong
  version. Guard: engine response carries `extra_fields["global_steps"]`
  (llm_server.py:417); discard version-mismatched results and re-issue in
  `on_sampled`.
- Pre-dispatched requests for keys that are ultimately not selected are aborted
  by `on_sample_end`; results are simply dropped. Harmless.

### Structure

```
sample() poll loop (each poll iteration)
  └─ for keys newly finished and not yet dispatched:
       issue re-prefill (async, non-blocking)
       store future in pending_prefill: {key: future}

on_sampled(batch)
  ├─ for each consumed key, look up pending_prefill:
  │    ├─ hit, version == W_{k-1} → use it (zero wait)
  │    ├─ hit, version mismatch   → re-issue, await
  │    └─ miss (old sample)       → issue now, await
  └─ await all futures, write new_rollout_log_probs to TQ
```

### Implementation mount points

- Pre-dispatch inside the poll loop: the replay buffer supports
  `trainer.v1.sampler.custom_sampler` injection (replay_buffer.py:408). Provide
  a subclass exposing an `on_new_finished(keys)` hook that the trainer registers
  its pre-dispatch callback into. (Alternative — a trainer-side background
  poller on TQ metadata — is simpler but less precise; the custom-sampler route
  is preferred for tighter timing.)
- Re-prefill issue path shared with P1 (§3.1 helpers), plus per-key future
  tracking (`pending_prefill: dict[str, asyncio.Task]`) with cleanup on
  selection/eviction.

### P2 acceptance

`new_rollout_log_prob` timer measurably reduced vs P1 (staleness_sweep's
existing timer name reused); no change in training metrics vs P1 (pipelining is
performance-only).

### Degenerate case

If all N samples were finished before sampling began, pre-dispatch has nothing
to overlap and P2 degrades to P1 synchronous behavior. Correct, just not faster.

## 6. P3 — separate async (out of scope here)

Same trainer mode on `trainer_separate_async`: re-prefill runs on the rollout
resource pool while the trainer trains — the largest real saving, and it
replaces the CPU save/restore dance entirely. Mechanism designed to be
deployment-agnostic; porting is a follow-up.

## 7. Risks

| Risk | Mitigation |
|---|---|
| T/R mismatch between engines makes re-prefill π_b biased vs trainer π_b | `mismatch/*` metrics already emitted; P1 arm 3 comparison quantifies it; acceptance is "no collapse", not bit-exactness |
| `on_sample_end` abort races in-flight re-prefill | P1: synchronous wait before return. P2: pre-dispatch only for keys whose results would be dropped if unselected |
| Re-prefill competes with in-flight generation on colocated GPUs | P1 acceptable; P2 overlaps it with generation wait; measure via `new_rollout_log_prob` timer |
| Prefix cache assumption (re-prefill of just-decoded trajectories is cheap) may not hold | Measure in P1 via timer; if cold, the saving argument weakens — decide then |
| Nested-jagged format regressions in downstream `to_padded_tensor()` | Reuse the exact storage format validated in staleness_sweep |

## 8. Testing

- Unit: `_slice_response_logprobs` off-by-one (prompt_len=1 edge), version
  guard logic, pending_prefill hit/miss/mismatch paths.
- Integration: colocate async small model (e.g. Qwen3-0.6B) — run
  `reprefill_decoupled` end-to-end; assert `old_log_probs == new_rollout_log_probs`
  in TQ; `staleness/*`, `mismatch/*`, `offpolicy/*` metrics present.
- Acceptance runs: three arms of §4 on a staleness-inducing config
  (large `num_warmup_batches` / buffer depth to force staleness).
