# Empty2D predicted-clean guidance conclusion

Status: closed after the seed-42 mechanism gate failed, 2026-08-10.

## Question and controlled setting

This experiment asked whether LAN-O3DP-style predicted-clean, thresholded
nearest-point guidance can turn collision-prone diffusion-policy rollouts into
task-completing safe rollouts when perception, robot geometry, and collision
checking are reduced to an idealized two-dimensional point setting.

The frozen policy observes only current XY and never observes the obstacle. It
outputs absolute XY targets, while Circle and Square boundary points are
available only to deployment-time guidance. The 90-trajectory training split
contains 33 Circle-safe and 20 Square-safe counterfactual routes, including
both sides of each obstacle, so absence of safe demonstrations is not a valid
explanation for the result.

## Source and calibration gates

The seed-42 sample-prediction model completed 500 epochs in required online
W&B run `ar7j7h06`. The fixed 50-seed source screen selected epoch 100 with
Task/Safe SR `1.00 / 1.00`, CR/NCR `0.00 / 0.00`, and zero execution clipping.

Circle and Square were independently calibrated on 20 fixed diffusion seeds
over all 20 preregistered combinations:

- `rho in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`;
- `Q in {0.02, 0.04, 0.08, 0.12}`.

Every candidate in both geometries produced the same task-level outcome:
Task SR `1.00`, Safe SR `0.00`, CR `1.00`, and NCR `0.00`. Guidance was active:
mean chunk trigger rate ranged from 15.0% to 39.8% for Circle and 20.4% to
40.5% for Square, but no candidate removed a collision. Because all candidates
tied, the preregistered lower-parameter tie break selected `rho=1e-4, Q=0.02`;
this is not evidence of a useful calibrated regime.

## Held-out paired result

Each geometry used 100 new diffusion seeds, paired exactly between baseline
and guided rollouts.

| Geometry | Condition | Task SR | Safe SR | CR | NCR | Success with collision |
|---|---|---:|---:|---:|---:|---:|
| Circle | baseline | 1.00 | 0.00 | 1.00 | 0.00 | 100/100 |
| Circle | guided | 1.00 | 0.00 | 1.00 | 0.00 | 100/100 |
| Square | baseline | 1.00 | 0.00 | 1.00 | 0.00 | 100/100 |
| Square | guided | 1.00 | 0.00 | 1.00 | 0.00 | 100/100 |

Both geometries have zero net rescues, zero CR reduction, and exact McNemar
`p=1.0`. Neither passes any component of the preregistered mechanism gate.

## Post-hoc scale diagnostics

Three single-seed Circle videos were produced after the formal grid to verify
that the gradient path was not inert:

- `rho=3e-4, Q=0.12`: guided and baseline paths are visually indistinguishable
  and both reach the goal with collision;
- `rho=0.1, Q=0.12`: guidance causes a visible deflection but still collides;
- `rho=1, Q=0.12`: the guided rollout stays collision-free but retreats or
  stalls and does not reach the goal, while baseline reaches through the
  obstacle.

These post-hoc examples show that sufficiently large guidance can change the
executed trajectory and enforce local clearance in one diagnostic rollout.
They do not constitute a safe-success regime or alter the formal negative
result.

The diagnostic behavior is consistent with a directional limitation. For the
active point cost `C(p)=[Q-||p-c*||]+`, descent moves directly away from the
selected obstacle point. In the centered head-on configuration, this direction
opposes goal progress and supplies no first-order preference between the upper
and lower bypass modes. Backpropagating the cost through the denoiser changes
the noisy sample, but it does not create the missing task-compatible tangential
direction. This is a mechanism hypothesis supported by the diagnostic, not a
separate preregistered causal test.

## Decision and claim boundary

Current guidance is not validated even in the simplified environments. The
planned seed-43 and seed-44 replications are not run because seed 42 neither
passes nor partially passes. Stop scaling or tuning this unchanged
nearest-point formulation.

The result does not show that predicted-clean gradients are numerically broken;
the trigger diagnostics and out-of-grid trajectory changes show otherwise. It
also does not support the stronger claim that complex swept gripper/object
geometry is the primary or unique cause of the earlier D2 failure. Instead, it
reveals an additional limitation already present in matched point geometry:
collision-only radial guidance can trade collision for task non-completion
without selecting a persistent bypass direction.

## Canonical artifacts

- source screen: `source_screen.json`;
- merged formal metrics: `eval_seed42/result.json`;
- generated formal report: `eval_seed42/report.md`;
- calibration summaries: `eval_seed42/calibration_{circle,square}_summary.json`;
- environment overview and three post-hoc scale-diagnostic videos:
  repository-local MP4 artifacts in this directory; visually redundant formal
  small-scale held-out videos were not retained;
- training record: W&B run `empty2d_guidance/ar7j7h06`.
