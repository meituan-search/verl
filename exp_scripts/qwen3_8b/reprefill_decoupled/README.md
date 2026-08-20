# Re-prefill Decoupled PPO — Qwen3-8B / Megatron acceptance harness

This directory contains the cluster launch script for the `reprefill_decoupled`
V1 PPO trainer mode and the acceptance-test arms the human runs on GPU.

`reprefill_decoupled` is a colocate-async PPO trainer in which `π_b`
(`old_log_probs`) is computed by **re-prefilling consumed trajectories on the
rollout engine at its current weight `W_{k-1}`**, replacing the trainer-side
`old_log_prob` forward pass. An optional P2 stage (pipelined pre-dispatch)
overlaps the re-prefill with remaining generation as samples finish during the
replay-buffer poll loop.

## Enable flags

All knobs are passed to the trainer config (`verl/trainer/config/ppo_trainer.yaml`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `trainer.use_v1=true` | — | Use the V1 PPO trainer entry point. |
| `trainer.v1.trainer_mode=reprefill_decoupled` | — | Select this trainer. |
| `trainer.v1.reprefill_decoupled.num_warmup_batches` | `1` | Number of warmup batches added before the training loop starts. Larger values deepen staleness (Arm 1 stress). |
| `trainer.v1.reprefill_decoupled.enable_prefill_pipeline` | `false` | `false` = **P1** (post-hoc re-prefill after the replay-buffer poll loop returns). `true` = **P2** (pipelined pre-dispatch, overlap re-prefill with remaining generation). |

The run script exposes the last two as env vars so P1 vs P2 comparison uses a
single script:

```bash
NUM_WARMUP_BATCHES=4 bash run_qwen3_8b_megatron_reprefill_decoupled.sh ...
ENABLE_PREFILL_PIPELINE=true bash run_qwen3_8b_megatron_reprefill_decoupled.sh ...
```

Rollout-correction presets are **not** baked into the script. They are passed
via `algorithm.rollout_correction.*` overrides on the command line (see the
arms below). The default `algorithm.rollout_correction` config group
(`verl/trainer/config/algorithm/rollout_correction.yaml`) has
`bypass_mode=false`, `rollout_is=null`, `rollout_rs=null` — i.e. correction
**disabled** (metrics-only). See `verl/trainer/config/algorithm.py`
(`RolloutCorrectionConfig`) for the full list of presets.

## The three acceptance arms (spec §4)

Acceptance criterion: **Arm 2 stable = success** (no entropy collapse, no KL
explosion, reward grows). Arms 1 and 3 are reference points for comparison.

### Arm 1 — negative control (expect collapse)

`reprefill_decoupled` with **bypass mode** (`algorithm.rollout_correction.bypass_mode=true`)
under **forced high staleness**. With bypass mode, the trainer skips the
trainer-side `old_log_prob` forward pass entirely and the PPO ratio becomes
`π_θ / π_rollout` against stale `π_rollout` — so under deep staleness this
should exhibit the classic off-policy failure: entropy collapse, KL
explosion, or reward stall.

Force staleness with either (or both):
- large `NUM_WARMUP_BATCHES` (e.g. `4`–`8`× the default `1`), and/or
- a deep replay buffer via `trainer.v1.sampler.max_off_policy_threshold`
  (default `8`; raise to e.g. `16`).

```bash
NUM_WARMUP_BATCHES=8 \
bash run_qwen3_8b_megatron_reprefill_decoupled.sh \
    algorithm.rollout_correction.bypass_mode=true \
    trainer.v1.sampler.max_off_policy_threshold=16
```

Expected: collapse (entropy collapse / KL explosion / reward stall).

### Arm 2 — the claim (expect stable, success)

`reprefill_decoupled` with the default (correction **disabled**, `bypass_mode=false`).
`π_b` comes from the rollout-side re-prefill at `W_{k-1}`, so the off-policy
mismatch that drives Arm 1 collapse should be removed — expect stable
training (no collapse; reward grows).

P1 (post-hoc re-prefill) is the default; P2 (pipelined pre-dispatch) is the
overlap variant — run both and compare the `new_rollout_log_prob` timer
(P2 should be faster than P1).

```bash
# P1 (default)
bash run_qwen3_8b_megatron_reprefill_decoupled.sh

# P2 (pipelined pre-dispatch)
ENABLE_PREFILL_PIPELINE=true bash run_qwen3_8b_megatron_reprefill_decoupled.sh
```

Expected: stable (no collapse; reward grows). This is the acceptance bar.

### Arm 3 — upper-bound reference (expect stable)

The existing **decoupled trainer-`π_b`** mode: `separate_async` trainer with
`algorithm.rollout_correction` in decoupled mode (`bypass_mode=false`) and a
decoupled IS preset. This is the three-policy decoupled trainer (π_θ, π_b,
π_rollout) where `π_b` is computed by a separate trainer-side forward pass —
the prior art `reprefill_decoupled` is replacing.

Run with the `staleness_sweep`/`colocate_async` launch scripts by selecting
`trainer.v1.trainer_mode=separate_async` and a decoupled IS preset:

```bash
# Token-level IS (decoupled_token_is preset)
bash ../staleness_sweep/run_qwen3_8b_megatron_staleness_sweep.sh \
    trainer.v1.trainer_mode=separate_async \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=token

# Sequence-level IS (decoupled_seq_is preset)
bash ../staleness_sweep/run_qwen3_8b_megatron_staleness_sweep.sh \
    trainer.v1.trainer_mode=separate_async \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=sequence
```

> Note: Arm 3 reuses the `staleness_sweep` launch script with overrides
> because that script already has the per-dataset defaults wired up. Only the
> `trainer_mode` and `rollout_correction` leaves need to change; the
> `staleness_sweep.num_steps` knob is ignored by `separate_async`. Adjust
> `trainer.v1.separate_async.num_warmup_batches` /
> `parameter_sync_step` as needed.

Expected: stable (this is the upper bound Arm 2 should match).

## Metrics to watch

Watch these in tensorboard / prometheus:

- `offpolicy/*` — the off-policy metric decomposition
  (`staleness/*`, `mismatch/*`, `combined/*`). The staleness bucket should be
  small and bounded in Arm 2; in Arm 1 it should be large and the mismatch
  should drive collapse.
- `staleness_sweep/sample_staleness_mean` and `staleness_sweep/sample_staleness_max`
  — per-sample staleness (model-version distance between `π_rollout` and the
  weight the sample was generated at). Useful for confirming the staleness
  stress in Arm 1 actually took effect.
- `reprefill_decoupled/resume_version` — the rollout-engine weight version
  (`W_{k-1}`) used for the re-prefill. Should step with `parameter_sync_step`.
- `reprefill_decoupled/old_log_prob_source` — confirms `old_log_probs` came
  from the rollout-side re-prefill (not a trainer-side forward pass).
- `new_rollout_log_prob` timer — the re-prefill wall-clock. **P1 vs P2
  comparison**: P2 (pipelined pre-dispatch) should be faster than P1
  (post-hoc) because it overlaps re-prefill with remaining generation.
- actor entropy, KL (`actor.kl_loss` / `actor.kl_coef`), reward — the
  training-health signals. Arm 2 must stay stable; Arm 1 should diverge.

## How to force staleness (Arm 1 stress)

- Raise `NUM_WARMUP_BATCHES` (e.g. `4`–`8`× the default `1`) so the replay
  buffer fills with samples generated at older weights before training starts.
- Raise `trainer.v1.sampler.max_off_policy_threshold` (default `8`) so the
  replay buffer accepts deeper-staleness samples instead of dropping them.
  (`max_off_policy_strategy=drop` by default; see
  `verl/trainer/config/ppo_trainer.yaml` `trainer.v1.sampler.*`.)

## Per-dataset defaults

The script picks per-dataset defaults via `DATASET` (`gsm8k` | `dapo`). Any
knob can still be overridden via its env var (`TRAIN_FILES`,
`TRAIN_BATCH_SIZE`, `ACTOR_LR`, `ROLLOUT_N`, `PPO_MAX_TOKEN_LEN_PER_GPU`,
`MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `ACTOR_TP`, `ACTOR_PP`,
`ROLLOUT_TP`, `ROLLOUT_GPU_MEM_UTIL`, `TOTAL_EPOCHS`, `SAVE_FREQ`,
`TEST_FREQ`, `PROJECT_NAME`, `EXPERIMENT_NAME`). See the script header for
the full list.

## Runtime environment

`runtime_env.yaml` carries the cluster env vars (proxy, NCCL, tensorboard dir,
`TransferQueue` pip dep). It mirrors `staleness_sweep/runtime_env.yaml` with
the tensorboard directory renamed to `..._reprefill_decoupled_t2`.
