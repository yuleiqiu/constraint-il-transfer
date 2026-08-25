# Tabletop DP baseline and guidance result

Status: the obstacle-free source baseline is valid. A targeted five-layout /
five-noise counterfactual evaluation finds 4/11 safe rescues and removes
physical contact in 10/11 eligible collision failures, but six of the eleven
become collision-free stalls. Guidance is mechanically and sometimes
task-level effective, not a robust avoidance solution.

The obstacle-free `PickPlaceCanTabletop` source task now has a trained,
loadable LAN-O3DP baseline using the 100-demo point-cloud / absolute-EEF
dataset. Demonstrations contain no obstacle and use direct transport from the
Can start region to the GoalTray.

## Selected model

- architecture: LAN-O3DP sample prediction, DDPM-100, horizons 2 / 16 / 8;
- data: 100/100 successful scripted demos, 25,599 transitions, 90/10 split;
- training: batch 256, seed 42, online W&B project
  `tabletop_guidance_baseline`, run `58wfbbrn`;
- selection: best 10-rollout clean screen at epoch 100 (`8/10`);
- checkpoint: `model_epoch_100_tabletop_can_pointcloud_abs_eef_100_success_0.8.pth`.

The checkpoint screen was non-monotonic:

| Epoch | 50 | 100 | 150 | 200 | 250 | 300 |
|---:|---:|---:|---:|---:|---:|---:|
| Clean Task SR (10 rollouts) | 0.70 | **0.80** | 0.40 | 0.50 | 0.60 | 0.50 |

Training loss continued to fall after epoch 100 while closed-loop performance
degraded. Training was stopped after epoch 301 and epoch 100 was retained by
rollout early stopping.

## Independent D0 validation

The selected checkpoint was evaluated for 50 clean rollouts at horizon 600,
with policy-noise seeds 26000--26049:

- Task SR: **35/50 = 0.70**;
- Wilson 95% interval: 0.562--0.809;
- successful rollout length: 205--534 steps (mean 367.8);
- two successful rollouts require more than 500 steps.

All actions were finite and executable; there were no simulator or action-
interface failures. The 15 task failures divide into:

- 4 failed or unstable grasps (Can remains on the source table);
- 6 post-lift stalls near the source;
- 2 transport/lower stalls above the goal;
- 3 successful placements without enough retreat to meet the inherited strict
  PickPlaceCan success condition.

This establishes a real closed-loop DP baseline for the controlled 3-D task,
but not a highly reliable one. Obstacle and guidance results must report both
absolute Task SR and paired transitions against this clean-source limitation.

## Obstacle and guidance result

Milk placed directly on the table is not a useful avoidance test: the carried
Can passes above it, giving the unguided policy 9/10 safe successes. Raising
Milk into the transport plane creates the intended interaction, but also makes
the task difficult: over ten raised-Milk layouts the baseline has Task SR
3/10, Safe SR 0/10, CR 8/10, and NCR 2/10.

The canonical paired comparison uses five fixed raised-Milk layouts (episodes
1, 4, 5, 7, and 8), `Q=0.08 m`, `rho=45`, a smoothed hinge, horizon 600, and
the exact same simulator states and diffusion seeds for baseline and guidance:

| Method | Task SR | Safe SR | CR | NCR |
|---|---:|---:|---:|---:|
| paired baseline | 1/5 | 0/5 | 3/5 | 2/5 |
| radial predicted-clean guidance | 1/5 | 1/5 | 1/5 | 3/5 |
| radial + smooth direct output tangent (max 8 mm) | 1/5 | 1/5 | 0/5 | 4/5 |

The direct tangent comparison contains one failure-to-success transition and
one success-to-failure transition. It removes collisions in this small fixed
set but converts most failures into collision-free non-completion; it does not
improve Task SR. A 4 cm direct tangent produces a visually strong single-state
safe success but causes an abrupt ejection-like trajectory and was not a
multi-layout result. The 8 mm smooth variant reduces that jump but returns to
stalling. Episode-4 success is also replay-sensitive: one rendered replay
stalls and a retry succeeds from the same nominal state and seed.

Therefore the supported conclusion is limited: the environment exposes the
desired avoidance geometry and direct output correction can force lateral
motion, but the current guidance formulations do not robustly preserve task
progress. This stage should remain closed unless a new mechanism addresses
handoff / recoverability rather than another scalar sweep of `rho`, `Q`, or
tangent magnitude.

The later counterfactual evaluation repeats five noise seeds on each of the
five layouts and conditions rescue on clean success plus matched obstacle
collision failure. Eleven of 25 triads qualify. Radial guidance creates four
safe rescues, all in layout 4; the other eligible outcomes are six
collision-free non-completions and one collision failure. Physical Milk
contact falls from 11/11 to 1/11. An exact open-loop audit confirms nonzero
gradients and centimeter-scale final target displacement, but moving away from
one selected point does not increase clearance to the full obstacle cloud.
See `counterfactual_rescue_5scene_5noise/conclusion.md`.

## Artifacts

- per-rollout result: `baseline_epoch100_eval50_h600_seed26000.json`;
- representative videos in `videos/`: episode 0 is a normal success, episode 7
  is a completed placement without retreat, episode 9 is a post-lift transport
  stall, and episode 23 is a failed grasp;
- W&B project: <https://wandb.ai/yuleiqiu-university-of-leeds/tabletop_guidance_baseline>;
- W&B run: <https://wandb.ai/yuleiqiu-university-of-leeds/tabletop_guidance_baseline/runs/58wfbbrn>.

Canonical obstacle artifacts:

- five-layout / five-noise counterfactual result, open-loop audits, and merged
  metrics: `counterfactual_rescue_5scene_5noise/`;
- raised-Milk baseline (10 layouts):
  `milk_raised_baseline_seed819_n10/summary.json`;
- fixed five-layout state bank and radial comparison:
  `raised_milk_rho45_multilayout_seed819_ep1_4_5_7_8/`;
- direct output-tangent paired result:
  `raised_milk_output_tangent_smooth01_multilayout_ep1_4_5_7_8/`;
- replay-sensitive safe-success visualization:
  `raised_milk_output_tangent_smooth01_ep4_visual_retry/`;
- representative safe-stall visualization:
  `raised_milk_output_tangent_smooth01_typical_ep4_ep7_visual/episode_007_rho_45_paired.mp4`.

See `ARTIFACT_INDEX.md` for the canonical / diagnostic / superseded directory
classification and `checkpoint_cleanup_candidates_2026-08-18.json` for the
non-selected checkpoint cleanup audit. The seven candidates in that audit
were deleted on 2026-08-18 after the selected epoch-100 hash and canonical
artifacts were verified.

Video rendering replays the saved layout and diffusion seed rather than a
recorded simulator state trajectory. The four representative videos preserve
their formal success/failure class. Episode 34 is retained separately in the
manifest as an informative sensitivity case: it was a slow success in the
formal run but a placement-without-retreat failure during rendered replay.
