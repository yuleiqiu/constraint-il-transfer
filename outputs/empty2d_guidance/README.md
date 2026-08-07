# Empty2D Guidance

Status: data and implementation gates pass; formal seed-42 source training is
running, 2026-08-07.

This experiment isolates whether the paper-aligned predicted-clean
point-trajectory gradient can produce task-level avoidance when the executed
robot and obstacle geometry are genuinely points in 2-D. It replaces the old
`toy_recovery` prototype rather than extending it.

## Frozen contract

- workspace `[-1,1]^2`, start `(-0.875,0)`, goal `(0.875,0)`;
- observation `agent_pos=[x,y]`; goal is fixed and implicit;
- absolute XY target action, clipped to 0.03 executed displacement;
- point radius 0.02, swept segment collision, goal tolerance 0.03, H320;
- centered radius-0.3 Circle or 0.6-by-0.6 Square at test time;
- obstacles never enter the policy observation and guidance receives exactly
  256 uninflated oracle boundary points.

The environment geometry is minimally derived from
`mpd-splines-public` commit
`3676cbf371b0e2a98663d8cef666f671e004e3da`; the upstream MIT license is
retained in `empty2d_guidance/MPD_LICENSE.txt`. No MPD SDF, spline, planner,
IsaacGym, or model code is included.

## Completed gates

The endpoint-conditioned RBF GP uses length scale 1.1, amplitude 8, seed 42,
and accepts 100 of 112 draws using only workspace and 0.03-step rejection.
Amplitude 8 is an explicit departure from the original value 1 to provide a
fair counterfactual-support test.

| Geometry | Safe train | Collision train | Upper safe | Lower safe |
|---|---:|---:|---:|---:|
| Circle | 33 | 57 | 19 | 14 |
| Square | 20 | 70 | 11 | 9 |

All endpoint, workspace, step, support, route-balance, collision-count, action
shift, and absorbing-terminal checks pass. The canonical manifest is
`data/manifest.json`; the ignored HDF5 SHA256 is
`6c613f9b60bb1204e2b4609a9e7aacc638d541be1b97fc4454dd6b5c4f95b061`.

The formal model has 65,267,202 parameters. A real batch-256 CUDA update has
finite deterministic loss `0.44514683`, peaks at 4.138 GiB allocated memory,
saves/reloads a checkpoint, and produces finite baseline and guided actions
with 100 finite reverse-step diagnostics. The full 500-epoch seed-42 run
started on GPU0 in `empty2d_guidance_sample_seed42/20260807210551`. Its
required online W&B run is `ar7j7h06`; checkpoint correspondence is recorded
in the run's local `logs/wandb_run.json`. Source screening, calibration, and
held-out testing remain gated on training completion.

## Result boundary

Even if Circle and Square both pass, the permitted conclusion is limited: the
guidance implementation and predicted-clean gradient work in idealized point
geometry. That supports—but cannot by itself prove—the hypothesis that D2
failed because an EEF-point cost mismatched swept gripper/object geometry. It
does not show that complex geometry is the only D2 limitation.
