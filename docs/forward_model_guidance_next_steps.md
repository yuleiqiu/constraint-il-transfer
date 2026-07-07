# Forward Model / Guidance Handoff

## Purpose

This document is the current handoff for the `Delta geo` part of the project:

```text
single-object OSC diffusion policy
  + external geometry information
  -> multi-object obstacle-aware execution
```

It should be enough for a new agent / new conversation to understand the
current conclusions and the next concrete experiment without replaying the
whole discussion history.

## Project Framing

The original goal is single-to-multi-object diffusion policy transfer. The
failure is decomposed into:

- `Delta vis`: visual ambiguity, i.e. which object is the target.
- `Delta geo`: physical obstruction, i.e. the target is known but the arm path
  collides with extra objects.

The current work assumes `Delta vis` is handled by oracle target mask / known
target information, and focuses on `Delta geo`.

**2026-07-06 status update**: this handoff remains useful for the
forward-model / action-ranking branch, but it is no longer the primary next
experiment. We found that built-in robosuite `OSC_POSE` can replay full-pose
absolute EEF targets when the action is constructed as
`[next_eef_pos, quat2axisangle(next_eef_quat_site), gripper]`, with controller
references refreshed after `reset_to`. The current primary Route B experiment
is to train a diffusion policy on that executable full-pose EEF action.

The near-term claim should not be:

```text
forward-model guidance works
```

The defensible claim is narrower:

```text
For OSC-controlled diffusion policies, naive action integration is an
unreliable trajectory proxy. A learned action-to-EEF forward model provides a
more accurate basis for geometry-aware action evaluation. The base policy does
sample geometry-safer chunks, but geometry-only action selection is not enough
to improve rollout success.
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

Current interpretation:

```text
current gradient-based guidance update: not reliable
forward model itself: still useful as a trajectory predictor
```

### 2. Route B Was Reopened With Full-Pose OSC Absolute Actions

The earlier Route B rejection was too broad. These built-in robosuite routes
still fail and remain useful negative results:

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
  Route B `image_v15_delta_eef.hdf5`
- mean final position error: 0.51 cm

Mink controller follow-up:

- Panda + WholeBodyMinkIK can replay expert full-pose absolute EEF targets.
- Full-pose replay with `ik_hand_ori_cost=0.05` gets 100% final success on the
  tested valid split and first 50 train demos.
- A learned full-pose EEF diffusion policy using that interface still gets
  0.0 rollout success. This does not rule out the corrected OSC full-pose
  interface, because it is a different controller and action convention.

Current Route B status:

```text
expert full-pose EEF replay through OSC absolute: passes
expert full-pose EEF replay through Mink IK: passes
learned full-pose EEF target policy through Mink IK: fails
learned full-pose EEF target policy through corrected OSC absolute: untested
```

Relevant docs:

```text
docs/route_b_validation/report.md
docs/route_b_validation/adapter_report.md
outputs/route_b_validation/panda_mink_controller/STAGE_CONCLUSION.md
```

### 3. Learned OSC Forward Model Is Useful

Forward model objective:

```text
f_hat(state_t, OSC action chunk[t:t+H]) -> actual future EEF xyz trajectory
```

Trained on:

```text
third_party/robomimic/datasets/can/yq/image_v15.hdf5
```

Config:

```text
configs/forward_model/osc_eef_forward_image_v15.json
```

Model output:

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

Interpretation:

- The forward model is substantially better than naive cumsum.
- Full-range random actions are out-of-distribution and produce larger
  absolute error.
- The model should be used as an action-chunk evaluator / trajectory predictor,
  not as proof that gradient guidance works.

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

## Current Code State

Robomimic subrepo:

```text
third_party/robomimic
branch: exp/guided-dp
```

Relevant commits:

```text
76e7fb4 feat: add forward-model trajectory backend
c203b21 feat: add parallel guided rollout script
```

Key robomimic files:

```text
third_party/robomimic/robomimic/utils/osc_forward_model_utils.py
third_party/robomimic/robomimic/utils/obstacle_guidance_utils.py
third_party/robomimic/robomimic/algo/guided_diffusion_policy.py
third_party/robomimic/robomimic/scripts/run_obstacle_guided_agent.py
third_party/robomimic/robomimic/scripts/run_obstacle_guided_agent_parallel.py
```

Key local scripts:

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

## Current Question

The primary question is now:

```text
Can diffusion policy learn the corrected executable full-pose EEF action and
roll it out closed-loop through OSC absolute mode?
```

Concrete next direction:

1. Build or reuse an action key with
   `[next_eef_pos, quat2axisangle(next_eef_quat_site), gripper]`.
2. Train DP with the same observation setup as the successful masked-image OSC
   policy, but with the corrected full-pose EEF action target.
3. Roll out through the same `OSC_POSE` absolute/world interface used by
   `docs/route_b_validation/playback_eef_pose.py`.

Concrete training decision recorded on 2026-07-06:

```text
start from clean image diffusion policy, not masked-image policy
action target = full-pose absolute OSC action, 7D
diffusion sampler = DDIM
num_train_timesteps = 100
num_inference_timesteps = 10
action normalization = min_max
```

The intended 7D action is:

```text
[
  next_obs/robot0_eef_pos,
  quat2axisangle(next_obs/robot0_eef_quat_site),
  actions[:, 6],
]
```

Important implementation checks before long training:

1. Dataset / checkpoint env metadata must select built-in `OSC_POSE` with
   `input_type=absolute`, `input_ref_frame=world`, and `kp=500`.
2. Axis-angle values must be normalized for DP training and unnormalized before
   `env.step`.
3. After unnormalization, the robomimic env path must not clip the absolute
   pose action back to `[-1, 1]`; absolute rotation components may exceed 1.
4. Dataset generation must use `robot0_eef_quat_site` in robosuite `[x, y, z, w]`
   order and must not reorder the quaternion before `quat2axisangle`.
5. If rollout uses `env.reset_to(...)`, refresh controller references / goals
   after reset as in `docs/route_b_validation/playback_eef_pose.py`.
6. Inference should not need online quaternion conversion if the learned action
   is already axis-angle; quaternion handling belongs in dataset creation.

Recommended validation order:

1. Generate the absolute EEF OSC dataset and replay expert actions through the
   corrected controller.
2. Replay the same actions through the robomimic env wrapper to catch action
   clipping or stale-controller-reset issues.
3. Run a debug / short training job and verify action dimension, normalization
   statistics, controller metadata in the checkpoint, and rollout action scale.
4. Only then run full PickPlaceCan training. Multi-object transfer should wait
   until the clean single-object policy succeeds with this action space.

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
- Do not revive the old `delta_eef_action -> OSC delta` Route B path.
- Do not treat the failed learned Mink EEF policy as evidence against the
  corrected built-in OSC full-pose route; that route is untested.
- Do not use the early 10-rollout scale sweep as final evidence; it is
  superseded by the larger controlled comparison.
- Do not move to point-cloud ranking before oracle-geometry ranking has a
  task-preserving scoring rule.

## Reading Order For A Fresh Agent

1. This file.
2. `docs/route_b_validation/report.md`
3. `docs/action_chunk_ranking_report.md`
4. `outputs/forward_model/osc_eef_forward_image_v15/summary.md`
5. `outputs/forward_model/random_rollout_validation/random_scale_0p3/SUMMARY.md`
6. `outputs/robomimic/eval/forward_model_controlled_comparison/SUMMARY.md`
7. `outputs/diagnostics/guidance_update_effect/SUMMARY.md`
8. `third_party/robomimic/AGENTS.md`
