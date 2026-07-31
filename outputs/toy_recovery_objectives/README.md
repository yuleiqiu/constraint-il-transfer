# Toy Recovery Objectives

This directory contains the independent PointNav rapid prototype artifacts.

Tracked artifacts should remain small: the frozen layout manifest, experiment
configuration, scale-selection audit, deterministic summaries, and conclusion
report. HDF5 trajectories, JSONL per-episode records, checkpoints, and videos
are ignored.

`implementation_verification.json` records the passing unit/smoke checks. The
corrected RGB-only source reused the repository's ResNet18 + SpatialSoftmax
stack and passed the formal calibration gate at D0/D1/D2 success
`1.00 / 0.54 / 0.06`. See `source_gate_report.md`.

The authorized 150-rollout latent-dynamics stage then passed its action-shuffle
test (`1.93x`) but did not beat latent persistence. The funnel stopped before
guidance or any test rollout. See `dynamics_gate_report.md`.
