# Empty2D predicted-clean guidance conclusion

Status: formal seed-42 H320 gate failed; fixed strong-guidance H600 successor
passed on offset layouts, updated 2026-08-24.

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
- `rho=1, Q=0.12`: the guided rollout stays collision-free through the
  original H320 evaluation window while the baseline reaches the goal through
  the obstacle. Continuing the guided rollout beyond H320 was visually
  observed to eventually reach the goal without collision, with substantial
  jitter near the avoidance route.

The `rho=1` observation is a qualitative mechanism-level positive result:
sufficiently strong predicted-clean guidance can produce both avoidance and
eventual task completion in matched 2-D point geometry. It also exposes the
cost of doing so: the path is slow and jittery enough to exceed the frozen
H320 budget. The extended continuation is not a preregistered or replicated
result, so it does not alter the numerical H320 tables or establish a robust
safe-success rate.

The diagnostic behavior is consistent with an efficiency and stability
limitation. For the active point cost `C(p)=[Q-||p-c*||]+`, descent moves
directly away from the selected obstacle point. In the centered head-on
configuration, this initially opposes goal progress and supplies no stable
first-order preference between the upper and lower bypass modes. Repeated
closed-loop denoising can nevertheless eventually realize a bypass at large
`rho`, but the observed jitter and additional horizon show that the route is
not selected efficiently. This is a mechanism hypothesis supported by the
diagnostic, not a separate preregistered causal test.

## Fixed long-horizon successor

The subsequent Circle experiment freezes twenty mirrored obstacle offsets and
their matched centered controls, uses `rho=1`, `Q=0.12`, and follows each
continuous rollout through H600 with H400/H500/H600 first-passage metrics.
Every direct start-to-goal segment remains obstructed.

At H600, both offset hinge variants reach Task/Safe SR 20/20 with zero
collision. Their centered counterparts reach 13/20; all seven remaining
trajectories are collision-free non-completions. The offset conditions are
already 19/20 at H400. Huberization reduces backward distance and direction
reversals but does not change the final success partition.

This is the replicated quantitative positive regime that the original
single-seed continuation suggested. It does not revise the preregistered H320
paper-scale tables; it answers a different, fixed strong-guidance and
long-horizon mechanism question. Canonical result:
`smooth_h600_circle/report.md`.

## Decision and claim boundary

The formal conclusion remains that paper-scale guidance does not pass the
fixed H320 paired gate. The H600 successor establishes that, in matched 2-D
point geometry, stronger guidance and a longer horizon can robustly produce
collision-free task completion when obstacle offsets break route symmetry.
The centered 13/20 control shows that the present radial cost remains
inefficient and unstable in a symmetric head-on configuration.

The contrast with D2 manipulation supports a geometry-and-temporal-complexity
explanation: D2 additionally requires swept gripper geometry, pose-aware
clearance, grasp preservation, and a longer task-compatible recovery. It does
not isolate those factors as the unique causal explanation. Do not interpret
the result as a reason for another blind `rho` / `Q` sweep; a successor should
target stability, route persistence, and whole-gripper geometry. The planned
seed-43 and seed-44 formal H320 replications were not run.

## Canonical artifacts

- source screen: `source_screen.json`;
- merged formal metrics: `eval_seed42/result.json`;
- generated formal report: `eval_seed42/report.md`;
- calibration summaries: `eval_seed42/calibration_{circle,square}_summary.json`;
- environment overview and three post-hoc scale-diagnostic videos, tracked as
  repository-local MP4 artifacts in this directory; the `rho=1` video records
  the collision-free H320 portion, while eventual longer-horizon arrival is a
  subsequent visual observation;
- training record: W&B run `empty2d_guidance/ar7j7h06`;
- fixed strong-guidance H400/H600 contracts, aggregates, reports, and
  representative videos: `smooth_h400_circle/` and `smooth_h600_circle/`.
