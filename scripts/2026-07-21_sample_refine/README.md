# Sample-Refine Experiment Tools

This directory is independent of the existing LAN-style guided-denoising
experiment. It loads the same frozen `diffusion_policy` checkpoint through the
in-memory `sample_refine_diffusion_policy` adapter.

## 1. Fixed-state parameter sweep

Use fixed states captured by `scripts/guided_denoising/same_state_diagnostic.py`:

```bash
uv run python scripts/2026-07-21_sample_refine/fixed_state_sweep.py \
  --states <fixed_states.hdf5> \
  --envs PickPlaceBreadCerealCan \
  --out-dir outputs/sample_refine/fixed_state_sweep
```

Parameter selection defaults to the two-distractor development environment;
three-distractor states must remain held out. The fixed grid is `K=4`,
refinement timestep 40, margins 2/3/4 cm, and scales
0.001/0.003/0.01. A configuration is eliminated for any action clipping, more
than 2 cm total predicted waypoint displacement from candidate 0, or less than
90% preservation of originally positive task progress. Eligible configurations
are ranked by actual short-prefix collision conversions, median predicted
clearance improvement (with differences below 1 µm treated as ties), smaller
margin, then smaller scale.

## 2. Paired development pilot

Run all four conditions from identical initial simulator states and candidate-0
noise:

```bash
MUJOCO_GL=egl uv run python scripts/2026-07-21_sample_refine/paired_rollout_eval.py \
  --envs PickPlaceBreadCerealCan \
  --seeds 0 1 2 \
  --n-rollouts 10 \
  --margin <selected-margin> \
  --guidance-scale <selected-scale> \
  --out-dir outputs/sample_refine/paired_pilot

uv run python scripts/2026-07-21_sample_refine/aggregate_results.py \
  outputs/sample_refine/paired_pilot
```

The conditions are baseline, sample only, refine only (`K=1`), and sample plus
refine. The report includes Task SR, Safe SR, CR, NCR, the four-way outcome
partition, synchronized inference overhead, candidate coverage, refinement
accept/fallback, and baseline-critical P0-regime stratification.

Only after the two-distractor pilot passes should the selected parameters be
held fixed and evaluated on `PickPlaceBreadCerealMilkCan` and `PickPlaceCan`.
