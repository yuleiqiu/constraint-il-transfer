# Corrected 50-layout guidance comparison

Authorized 2026-09-09: 100 full-episode rollouts, each recorded during execution.
Completed and reviewed 2026-09-10; see the output directory's `conclusion.md`.
Question: does fixed original point guidance improve safe completion relative
to the fixed wide-support epoch-50 source on the entire seed-828 RaisedMilk bank?
No outcome-based selection, new model, training, teacher intervention, or tuning.

## Fixed protocol

- Episodes 0--49, H600, original policy seeds `828000000 + episode`.
- Source and original point guidance: rho45, Q0.08, Huber smoothing0.01,
  XY only, no tangent, active after Can height exceeds table height +0.08m.
- Same original `paired_eval.rollout_condition` and action contract; full pose
  absolute world EEF actions, complete collision physics, stop on true success
  or H600 (also preserve environment-done handling), never stop on collision.
- Reset XML/state and explicitly synchronize Python goal/obstacle XY with bodies.
  Seed raw environment Generator plus Python/NumPy/Torch sampling. Materialize
  the scene before the seeded evaluation reset. Audit initial full observations.
- GPU0: layouts0--24; GPU1:25--49. Each layout's pair stays in the same process.
  Initial two-condition pairs at layouts0/25 repeat twice with videos, then wait
  for a shared `START_MAIN` file. These eight preflight rollouts are separate
  from the requested 100. Require exact repeated states, actions, observation
  hashes, chunk cloud/RNG hashes and outcomes before creating the marker.
- Main layouts0/25 also compare against their preflight recordings. Both cards
  must pass their own checks; no cross-GPU determinism claim is made.

## Outputs and recording

Canonical output: `outputs/tabletop_guidance_reeval_2026-09-09/`.
Each shard has manifest, preflight gate and completion marker. Each main layout
has baseline/guided subdirectories and pair audit. Each branch contains:

- `rollout.mp4`: agentview384px with condition/layout/step and task/contact/
  goal/retreat labels; 10fps, normally every2 control steps, including terminal
  frame. Actual frame-step indices are in trajectory NPZ (terminal frames can
  introduce a half-frame timing discrepancy). Partial video name until closed.
- `trajectory.npz`: all executed actions, initial and every subsequent dynamic
  simulator state, video-step indices. The original state-bank XML is retained.
- `observations.npz`: full initial observation, every chunk's full observation
  history and obstacle cloud; no pickle required to read these arrays.
- `trace.json`: per-step EEF/Can/goal positions, actual predicate components,
  cumulative contact/task status, environment done, full observation hashes.
- `chunks.json`: chunk boundaries, guidance activation and cloud/RNG hashes.
- `reset.json`: before/after metadata and restored-state/initial-observation audit.
- `result.json`: final metrics, stop reason, seeds, artifact location and runtime;
  written last, serving as the branch completion record.

Models/state banks are referenced and hashed, not copied. Historical outputs
are preserved. Large arrays/videos remain local ignored artifacts; final reports,
JSON metrics and Markdown navigation are reviewable. No git commit or root AGENTS
edit is implied. Do not overwrite or reuse partial output directories silently.

## Execution

Use the runtime in `docs/runtime_setup.md`; inspect competing workloads first.
Each command is launched in a separate process with its own log:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 ROBOMIMIC_GPU_ID=0 \
uv run python -u scripts/2026-09-09_tabletop_guidance_reeval/run.py \
  --out outputs/tabletop_guidance_reeval_2026-09-09 --start 0 --stop 25

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 ROBOMIMIC_GPU_ID=0 \
uv run python -u scripts/2026-09-09_tabletop_guidance_reeval/run.py \
  --out outputs/tabletop_guidance_reeval_2026-09-09 --start 25 --stop 50
```

After inspecting both `preflight_pass.json` files, create `START_MAIN`.
Run `analyze.py --out outputs/tabletop_guidance_reeval_2026-09-09` for final
validation, aggregate report and the video review index. `--partial` permits
provisional progress output and must not be presented as the final estimate.

## Task-level gate and interpretation

Report Task SR, Safe SR, CR, NCR, four-way partition and paired rescue/regression
counts. A positive claim needs supported Safe SR improvement with task loss
explicitly assessed; an inconclusive difference is not proof of no effect.
The same historical 50 states constitute a corrected fixed-bank evaluation,
not new held-out validation for any future method. Eight identical-seed
preflight repetitions add no independent layout samples. No clean-environment
success claim or always-on comparison is made by this 100-branch protocol.
