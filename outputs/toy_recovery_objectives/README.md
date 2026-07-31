# Toy Recovery Objectives

This directory contains the independent PointNav rapid prototype artifacts.

Tracked artifacts should remain small: the frozen layout manifest, experiment
configuration, scale-selection audit, deterministic summaries, and conclusion
report. HDF5 trajectories, JSONL per-episode records, checkpoints, and videos
are ignored.

`implementation_verification.json` records the passing unit/smoke checks. The
source was retrained from scratch with absorbing zero-delta terminal padding,
using the identical 200-demo HDF5. The new checkpoint passes the unchanged
calibration gate at D0/D1/D2 success `1.00 / 0.56 / 0.14` and supersedes the
repeated-terminal source for future Toy work. See
`source/absorbing_terminal/report.md`.

The earlier authorized 150-rollout latent-dynamics stage used the superseded
source encoder. It passed its action-shuffle test (`1.93x`) but did not beat
latent persistence and must not be reused with the absorbing source. Its small
audit artifacts are preserved under `archive/repeated_terminal_source/`; the
raw rollout HDF5, model, and five legacy source checkpoints were deleted.

Before the absorbing-source repair, an explicitly requested geometry-only
diagnostic bypassed the
irrelevant latent-dynamics dependency while retaining the frozen policy,
manifest, oracle obstruction trigger, and screen contract. Scale 5 changed
screen SR from source to geometry by D0/D1/D2
`1.00/0.53/0.10 -> 1.00/0.58/0.16`. It reduced collisions but reached only
`+5/100` and `+6/100` net paired transitions in D1 and D2, below the
preregistered `+10/100` screen threshold. It is historical pre-fix evidence;
scale selection and screen outcomes cannot be reused for the new source. See
`archive/repeated_terminal_source/geometry_guidance/report.md`.

Current canonical layout and source assets:

- `layout_manifest.json`: the sole frozen calibration/tuning/screen/test
  layout manifest; duplicate candidate and gate copies were removed after
  byte-identity verification;
- `data/d0_demonstrations.hdf5`: the retained 200-demo source dataset;
- `train/source_seed0_absorbing_v3/source_update_001000.pth` through
  `source_update_005000.pth`: the five retained source checkpoints needed by a
  fresh dynamics collection;
- `visualizations/d{0,1,2}_oracle_vs_source.gif`: the retained user-facing
  oracle/source comparisons.

No dynamics rollout has been collected from the absorbing source, and no
screen or test layout has been evaluated with it. A future latent experiment
must start by recollecting the fixed 150-rollout dynamics dataset from the five
retained checkpoints.
