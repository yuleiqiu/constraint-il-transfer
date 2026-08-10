# Empty2D Guidance

Status: experiment closed after both seed-42 mechanism gates failed,
2026-08-10. See `conclusion.md`.

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
with 100 finite reverse-step diagnostics. The full seed-42 run completed all
500 epochs in `empty2d_guidance_sample_seed42/20260807210551`; required online
W&B run `ar7j7h06` contains the training record and its local manifest maps all
five formal checkpoints. The fixed 50-seed D0 screen selected epoch 100 with
Task/Safe SR `1.00`, CR/NCR `0.00 / 0.00`, and zero action clipping, so all
source gates pass. Its canonical result is `source_screen.json`.

## Final result

All 20 calibration settings in both geometries retain Task SR `1.00` but have
Safe SR `0.00` and CR `1.00`. The selected setting therefore reflects only the
lower-parameter tie break. On 100 new paired seeds per geometry, baseline and
guided rollouts again reach the goal with collision in every pair: zero net
rescues, zero CR reduction, and exact McNemar `p=1.0`.

Post-hoc Circle diagnostics show that much larger scales make the intervention
visible: `rho=0.1` deflects but still collides, while `rho=1` avoids collision
in one rollout by retreating or stalling and fails to reach the goal. This
confirms that the gradient path can affect execution but does not establish a
safe-success regime. Seed 42 failed rather than partially passed, so seeds 43
and 44 were not trained. The canonical interpretation and artifact index are
in `conclusion.md`.
