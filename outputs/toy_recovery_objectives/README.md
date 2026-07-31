# Toy Recovery Objectives

This directory contains the independent PointNav rapid prototype artifacts.

Tracked artifacts should remain small: the frozen layout manifest, experiment
configuration, scale-selection audit, deterministic summaries, and conclusion
report. HDF5 trajectories, JSONL per-episode records, checkpoints, and videos
are ignored.

`implementation_verification.json` records the passing unit/smoke checks and a
100-layout-per-difficulty oracle sample. The formal 200-demo source run was
then executed through its fixed 10,000-update cap, but failed the calibration
gate at D0 success `0.76 < 0.95`. Per the preregistered funnel, dynamics
collection and every guidance experiment were not run. See
`source_gate_report.md`.
