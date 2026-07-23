# Task-Constrained Guidance Experiment

This directory implements the staged candidate in
`docs/guided_denoising/task_constrained_posterior_guidance_plan.md`.

Stage 1 is complete and did not pass its preregistered 20-layout gate. The
posterior-gradient Stage 2 was intentionally not implemented. Canonical result:
`outputs/task_constrained_guidance/stage1_report.md`.

Stage 1 keeps the frozen policy's `observation / prediction / action` horizons
at `2 / 16 / 8`. It applies an oracle, task-progress-constrained XY correction
to the existing eight executable delta-EEF actions. Orientation, Z, and gripper
actions are unchanged.

Run the CPU-only geometry and optimization tests with:

```bash
uv run python scripts/2026-07-22_task_constrained_guidance/test_feasibility.py
```

Evaluate the fixed saved collision states with the frozen `2 / 16 / 8`
checkpoint contract:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-22_task_constrained_guidance/fixed_state_feasibility.py
```

For a state captured one chunk before its first collision, `--continuation-chunks
1` executes the supplied baseline/corrected chunk and then one normally
resampled eight-action chunk. Baseline and corrected branches use the same
continuation noise seed; this is ordinary receding-horizon evaluation and does
not change any policy horizon.

The original fixed-state bank contains late states whose chunk-start EEF point
can already lie inside an XY footprint, and it mixes collisions not explained
by the EEF-circle proxy. Capture the specific pre-entry regime with:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-22_task_constrained_guidance/capture_preentry_states.py \
  --minimum-start-clearance 0.01 \
  --predicted-clearance-threshold 0.01 \
  --minimum-collision-lead-chunks 1 \
  --max-collision-lead-chunks 1
```

The optimizer is an upper-bound diagnostic, not the final guided-denoising
method. Its retained scripts reproduce the stopped Stage 1 result; they should
not be treated as a deployable guidance method.
