# PointNav Recovery-Objective Rapid Prototype

Status: source gate failed on 2026-07-31. The formal 200-demo source run used
its fixed 10,000-update cap and reached D0 calibration success `0.76 < 0.95`.
The funnel therefore stopped before dynamics collection or guidance. See
`outputs/toy_recovery_objectives/source_gate_report.md`.

## Scope

This is an independent mechanism-isolation prototype. The source policy is
trained only on 200 obstacle-free D0 demonstrations. D0/D1/D2 rollouts may
train the action-conditioned latent predictor, without reward, correction, or
outcome labels. They never enter policy training. The global comparator is
named **global latent-manifold guidance (LPB-style objective)**; it is not a
full LPB reproduction.

The immutable contracts live in `toy_recovery/`:

- circular 2D agent, clipped delta-XY actions, 80-step common horizon;
- seeded template layouts and a safe-margin A* oracle;
- exact state, 64x64 RGB, exact clone/restore;
- HDF5 trajectory and JSONL evaluation schemas;
- source `ConditionalUnet1D` with fixed `2 / 16 / 8` horizons and DDIM
  `100 / 10`, initially trained for 5,000 updates with EMA checkpoints at each
  1,000 updates;
- shared predicted-clean guidance sampler and shared oracle obstruction
  trigger;
- geometry clearance-plus-progress and global D0 latent-bank objectives;
- 6/8 frozen-source continuation definition of safe recoverability.

## Order of execution

All commands run from the repository root.

```bash
uv run python scripts/2026-07-30_toy_recovery_objectives/generate_data.py \
  --config scripts/2026-07-30_toy_recovery_objectives/rapid_config.json

uv run python scripts/2026-07-30_toy_recovery_objectives/train_source.py \
  --config scripts/2026-07-30_toy_recovery_objectives/rapid_config.json \
  --data outputs/toy_recovery_objectives/data/d0_demonstrations.hdf5 \
  --output-dir outputs/toy_recovery_objectives/train/source_seed0
```

Run `environment_gate.py` against the candidate manifest and final source
checkpoint. Only a passing calibration result writes the frozen
`layout_manifest.json`; the command never evaluates source behavior on the
test split.

If and only if the 5k checkpoint misses the D0 retention gate while checkpoint
performance is still improving, one optimizer/online/EMA resume to the fixed
10k cap is allowed. This decision uses D0 only; D1/D2 outcomes may not select
the source checkpoint.

Pass all five ordered source checkpoints to `collect_dynamics.py`, then train
the predictor and D0-only bank:

```bash
uv run python scripts/2026-07-30_toy_recovery_objectives/collect_dynamics.py \
  --config scripts/2026-07-30_toy_recovery_objectives/rapid_config.json \
  --checkpoints <ckpt1> <ckpt2> <ckpt3> <ckpt4> <ckpt5> \
  --output outputs/toy_recovery_objectives/data/dynamics_seed0.hdf5

uv run python scripts/2026-07-30_toy_recovery_objectives/train_dynamics.py \
  --config scripts/2026-07-30_toy_recovery_objectives/rapid_config.json \
  --source-checkpoint <final-source-checkpoint> \
  --rollouts outputs/toy_recovery_objectives/data/dynamics_seed0.hdf5 \
  --d0-demos outputs/toy_recovery_objectives/data/d0_demonstrations.hdf5 \
  --output-dir outputs/toy_recovery_objectives/train/dynamics_seed0
```

The dynamics collection is fixed at 150 rollouts: ten for every D0/D1/D2 and
source-checkpoint combination. At each main-rollout replan state, the exact
predicted 16-action chunk is executed open loop from a simulator clone; this
matches the chunk consumed by guidance. Train/validation splitting is by
rollout and stratified by difficulty and checkpoint. `train_dynamics.py`
requires both shuffled-action MSE at least `1.2x` normal MSE over ten fixed
permutations and improvement over the latent-persistence baseline. Do not run
guidance after either failure.

The frozen manifest contains disjoint `calibration` (50/D), `scale_tuning`
(30/D), `screen` (100/D), and untouched `test` (200/D) splits. Scale selection
is permitted only with `evaluate.py --split scale_tuning --tune-scales`.
Run paired evaluation with `--split screen`, then run
`fixed_state_diagnostic.py` on source failures from that same screen split before
applying the rapid gate with `analyze_screen.py`. The analyzer exits nonzero
when the screen fails; confirmation and any ReGuide-style extension must not
run in that case.

## Gates

Before freezing the candidate layout manifest:

- oracle D0/D1/D2 success at least 0.99;
- source D0 at least 0.95;
- source D1 in `[0.20, 0.80]`;
- source D2 in `[0.05, 0.50]` and at least 0.10 below D1.

The pilot recovery diagnostic uses 10 failure states per D1/D2, 16 shared
source-pool candidates per state, four paired-noise guided candidates per
method, and the unchanged 6/8 continuation definition. It
reports objective ranking on the shared source candidate pool separately from
the recoverable-state coverage actually reached by each guided sampler. Expand
to 25 states/D and 32 candidates only after a positive pilot.

The rapid screen advances only if D0 has exact source parity with zero guidance
activations, at least one
guide has `F->S - S->F >= 10/100` in D1 or D2, and recovery ranking beats
random. Otherwise archive the boundary and do not implement ReGuide.
After a pass, repeat source/dynamics training with the three confirmation seed
pairs in `rapid_config.json` and evaluate each frozen method on the 200
untouched test layouts per difficulty. The same selected tuning scales
and common 80-step horizon apply to every seed.

For a cheap implementation check:

```bash
uv run python scripts/2026-07-30_toy_recovery_objectives/smoke_test.py
```
