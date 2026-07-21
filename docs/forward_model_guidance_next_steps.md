# Forward Model / Guidance Archive

**Status (2026-07-08)**: archived result document. The forward-model,
obstacle-guidance, and action-ranking implementation code has been removed from
the active repository. The active replacement is `delta_eef_pose_action`, whose
policy chunks reconstruct executed EEF pose trajectories directly; see
`outputs/eef_pose_osc_policy/README.md`.

## Purpose

This document records the completed `Delta geo` forward-model / ranking branch:

```text
single-object OSC diffusion policy
  + external geometry information
  -> multi-object obstacle-aware execution
```

It is retained so the experiment outcome remains auditable without keeping the
implementation code.

## Project Framing

The original goal is single-to-multi-object diffusion policy transfer. The
failure is decomposed into:

- `Delta vis`: visual ambiguity, i.e. which object is the target.
- `Delta geo`: physical obstruction, i.e. the target is known but the arm path
  collides with extra objects.

The archived work assumed `Delta vis` was handled by oracle target mask / known
target information, and focused on `Delta geo`.

**2026-07-08 status update**: the active replacement is delta EEF pose policy.
It learns an executable full-pose EEF action and reconstructs executed EEF pose
chunks directly. The old OSC-action forward model is no longer part of the
active code path.

The near-term claim should not be:

```text
forward-model guidance works
```

The defensible claim is narrower:

```text
For original OSC-action diffusion policies, naive action integration is an
unreliable trajectory proxy. A learned action-to-EEF forward model was a better
diagnostic predictor, but geometry-only action selection was not enough to
improve rollout success. Delta EEF pose replaces this route by making
action-chunk to EEF-pose trajectory reconstruction direct.
```

## Confirmed Facts

### 1. Original Gradient Guidance Did Not Improve Success

The original obstacle guidance used:

```text
EEF trajectory ~= current_eef + cumsum(action[..., :3] * 0.05)
```

Large-scale rollout results showed no success-rate improvement. This was not
just a tuning issue: the trajectory proxy has centimeter-level error under the
OSC controller, comparable to obstacle radii.

Relevant outputs:

```text
outputs/robomimic/eval/forward_model_controlled_comparison/SUMMARY.md
outputs/diagnostics/guidance_update_effect/SUMMARY.md
```

Archived interpretation:

```text
current gradient-based guidance update: not reliable
delta_eef_pose_action replaces the OSC-action forward-model route
```

### 2. EEF Action-Interface Validation Was Resolved

The earlier rejection of EEF-space control was too broad. These tested routes
failed and remain useful negative results:

- `delta_eef_action -> OSC delta`: fails replay.
- position-only `next_eef_pos -> OSC absolute`: fails replay.
- cumulative EEF delta -> built-in IK delta: fails replay.
- `real delta_EEF -> OSC command` adapters: better offline regression but
  still fail open-loop replay.

Corrected built-in OSC route:

- `abs_eef_pose_action = [next_eef_pos, quat2axisangle(next_eef_quat_site), gripper]`
- controller: built-in `OSC_POSE`, `input_type=absolute`, `input_ref_frame=world`,
  `kp=500`
- replay result: 200/200 final success on both `image_v15.hdf5` and the older
  `image_v15_delta_eef.hdf5`
- mean final position error: 0.51 cm

Mink controller follow-up:

- Panda + WholeBodyMinkIK can replay expert full-pose absolute EEF targets.
- Full-pose replay with `ik_hand_ori_cost=0.05` gets 100% final success on the
  tested valid split and first 50 train demos.
- A learned full-pose EEF diffusion policy using that interface still gets
  0.0 rollout success.

Final interpretation:

```text
expert full-pose EEF replay through OSC absolute: passes
expert full-pose EEF replay through Mink IK: passes
learned full-pose EEF target policy through Mink IK: fails
clean-image absolute EEF policy through OSC: 0.82 rollout success
clean-image delta EEF policy through OSC: 0.98 rollout success
```

The delta EEF policy is the preferred interface and reconstructs its executed
trajectory directly. The superseded controller-validation implementation and
per-controller outputs were removed from the active tree; their audit trail is
preserved in Git history and `docs/RESEARCH_LOG.md`. Canonical current results
are in `outputs/eef_pose_osc_policy/README.md` and
`docs/eef_pose_osc_policy_training.md`.

### 3. Learned OSC Forward Model Was a Useful Diagnostic

Forward model objective:

```text
f_hat(state_t, OSC action chunk[t:t+H]) -> actual future EEF xyz trajectory
```

Trained on:

```text
third_party/robomimic/datasets/can/yq/image_v15.hdf5
```

Historical config, now removed from active code:

```text
configs/forward_model/osc_eef_forward_image_v15.json
```

Historical model output, now removed from active outputs:

```text
outputs/forward_model/osc_eef_forward_image_v15/model.pth
```

Held-out demo validation:

| predictor | traj RMSE | terminal mean |
|---|---:|---:|
| learned forward model | 0.264 cm | 0.280 cm |
| `cumsum(action * 0.05)` | 11.831 cm | 17.560 cm |
| best fitted linear cumsum | 0.857 cm | 0.917 cm |

Random-action env rollout validation:

| setting | model traj RMSE | cumsum traj RMSE |
|---|---:|---:|
| random action scale 1.0 | 6.56 cm | 13.73 cm |
| random action scale 0.3 | 2.72 cm | 4.01 cm |

Archived interpretation:

- The forward model is substantially better than naive cumsum.
- Full-range random actions are out-of-distribution and produce larger
  absolute error.
- The result explains why the original OSC action interface was a poor target
  for geometry guidance.
- The model is no longer an active dependency because `delta_eef_pose_action`
  gives the policy a directly reconstructible EEF pose chunk.

Relevant outputs:

```text
outputs/forward_model/osc_eef_forward_image_v15/summary.md
outputs/forward_model/random_rollout_validation/full_random_scale_1p0/SUMMARY.md
outputs/forward_model/random_rollout_validation/random_scale_0p3/SUMMARY.md
```

### 4. Action-Chunk Ranking Found Safe Samples But Did Not Improve Success

The immediate action-ranking question has been tested:

```text
Does the single-object policy sample geometry-safe action chunks?
```

Answer:

```text
Yes, but geometry-only ranking is not a successful rollout method.
```

Same-state diagnostic:

| backend | K | actual clearance improved | mean actual clearance delta |
|---|---:|---:|---:|
| forward model | 16 | 8/10 | +1.54 mm |
| cumsum | 16 | 4/10 | -0.74 mm |

Hardest-environment rollout pilot:

```text
PickPlaceBreadCerealMilkCan
seeds = 700, 701, 702
20 rollouts / seed
```

| condition | success | collision-any | collision steps |
|---|---:|---:|---:|
| no guidance | 53/60 = 0.883 | 0.417 | 23.63 |
| forward-model ranking K=4 | 35/60 = 0.583 | 0.383 | 47.27 |
| forward-model ranking K=16 gated | 50/60 = 0.833 | 0.367 | 36.90 |

Interpretation:

- The policy distribution contains safer chunks.
- The forward model is a better selector than `cumsum` in same-state tests.
- Selecting only for obstacle clearance can choose off-task chunks and hurt
  grasp / placement timing.
- The oracle-mask baseline is already high in the hardest environment, so
  remaining `Delta geo` success headroom is smaller than expected.

Relevant doc:

```text
docs/action_chunk_ranking_report.md
```

## Archived Code State

The following paths are historical references from the completed branch. The
implementation files have been removed from the active repository.

Robomimic subrepo at the time:

```text
third_party/robomimic
branch: exp/guided-dp
```

Relevant commits:

```text
76e7fb4 feat: add forward-model trajectory backend
c203b21 feat: add parallel guided rollout script
```

Removed robomimic files:

```text
third_party/robomimic/robomimic/utils/osc_forward_model_utils.py
third_party/robomimic/robomimic/utils/obstacle_guidance_utils.py
third_party/robomimic/robomimic/algo/guided_diffusion_policy.py
third_party/robomimic/robomimic/scripts/run_obstacle_guided_agent.py
third_party/robomimic/robomimic/scripts/run_obstacle_guided_agent_parallel.py
```

Removed local scripts:

```text
scripts/train_osc_eef_forward_model.py
scripts/check_osc_forward_model_grad.py
scripts/diagnose_control_timing.py
scripts/diagnose_forward_model_random_rollout.py
scripts/diagnose_guidance_update_effect.py
scripts/2026-07-03_action_chunk_ranking/run_ranking_diagnostic.py
scripts/2026-07-03_action_chunk_ranking/run_ranking_eval_matrix.py
scripts/2026-07-03_action_chunk_ranking/aggregate_ranking_results.py
```

Important checkpoint:

```text
outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth
```

## Superseded Question

The primary question was whether a diffusion policy could learn an executable
full-pose EEF action and roll it out closed-loop through OSC. It has been
answered:

```text
corrected absolute EEF replay through OSC: passes
clean-image absolute EEF policy: 0.82 PickPlaceCan rollout success
clean-image delta EEF policy: 0.98 PickPlaceCan rollout success
delta EEF trajectory reconstruction: 0.131 cm mean position error
```

No further controller-route experiment is needed. Current work should keep the
validated `delta_eef_pose_action` interface and focus on deployment costs that
preserve task intent during guided denoising. See
`outputs/eef_pose_osc_policy/README.md` for the conclusion and
`docs/eef_pose_osc_policy_training.md` for the maintained pipeline.

The action-ranking experiment remains a completed secondary branch. It changed
that branch's bottleneck: the question is no longer whether safe chunks exist;
they do. If ranking is revisited, the question should be:

```text
Can action selection preserve task intent while avoiding obstacles?
```

Concrete ranking directions:

1. Add task-preserving ranking terms:
   - obstacle cost as a hard filter;
   - distance to the first sampled chunk as a regularizer / tie-break;
   - policy likelihood or denoising score as a prior;
   - task-progress heuristics near grasp / place phases.
2. Re-check failure attribution:
   - success is already high with no guidance;
   - collision-any is not tightly coupled to failure;
   - determine whether failed episodes are actually caused by non-target
     collision, grasp timing, placement, or other policy errors.
3. Only after ranking is reliable with oracle geometry, revisit point-cloud
   geometry as a replacement for oracle obstacle centers.

## What Not To Do Next

- Do not claim forward-model guidance works based on forward-model validation.
- Do not keep tuning geometry-only ranking; it has been tested and hurts
  rollout success.
- Do not revive the old `delta_eef_action -> OSC delta`, adapter, or Mink
  controller paths; the delta EEF pose interface has replaced them.
- Do not use the early 10-rollout scale sweep as final evidence; it is
  superseded by the larger controlled comparison.
- Do not move to point-cloud ranking before oracle-geometry ranking has a
  task-preserving scoring rule.

## Reading Order For A Fresh Agent

1. This file.
2. `outputs/eef_pose_osc_policy/README.md`
3. `docs/eef_pose_osc_policy_training.md`
4. `docs/action_chunk_ranking_report.md`
5. `outputs/forward_model/osc_eef_forward_image_v15/summary.md`
6. `outputs/forward_model/random_rollout_validation/random_scale_0p3/SUMMARY.md`
7. `outputs/robomimic/eval/forward_model_controlled_comparison/SUMMARY.md`
8. `outputs/diagnostics/guidance_update_effect/SUMMARY.md`
9. `third_party/robomimic/AGENTS.md`
