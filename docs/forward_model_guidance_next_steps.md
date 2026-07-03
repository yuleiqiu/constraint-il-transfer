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

The near-term claim should not be:

```text
forward-model guidance works
```

The defensible claim is narrower:

```text
For OSC-controlled diffusion policies, naive action integration is an
unreliable trajectory proxy. A learned action-to-EEF forward model provides a
more accurate basis for geometry-aware action evaluation. Whether this enables
single-to-multi transfer depends on whether the base policy samples
geometry-safe action chunks.
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

### 2. Route B, Predicting EEF Instead of OSC Action, Is Not Solved

Built-in robosuite controller routes fail:

- `delta_eef_action -> OSC delta`: fails replay.
- `next_eef_pos -> OSC absolute`: fails replay.
- cumulative EEF delta -> built-in IK delta: fails replay.
- `real delta_EEF -> OSC command` adapters: better offline regression but
  still fail open-loop replay.

Mink controller follow-up:

- Panda + WholeBodyMinkIK can replay expert full-pose absolute EEF targets.
- Full-pose replay with `ik_hand_ori_cost=0.05` gets 100% final success on the
  tested valid split and first 50 train demos.
- A learned full-pose EEF diffusion policy using that interface still gets
  0.0 rollout success.

Current Route B status:

```text
expert EEF replay through Mink IK: passes
learned EEF target policy through Mink IK: fails
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
```

Important checkpoint:

```text
outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth
```

## Immediate Question

Before further tuning gradient guidance, answer:

```text
Does the single-object diffusion policy generate any geometry-safe action
chunks under multi-object obstruction?
```

If such chunks exist, use the forward model to select them. If they do not
exist, inference-time correction alone is unlikely to solve single-to-multi
transfer, and the project needs training-side adaptation / new data
distribution.

## Proposed Next Experiment: Action-Chunk Ranking

Replace gradient guidance with sampling and ranking:

1. At each decision point, sample multiple candidate action chunks from the
   same diffusion policy.
2. Predict each candidate's EEF trajectory with the learned forward model.
3. Score each trajectory using obstacle geometry.
4. Execute the safest candidate, without applying gradient updates to the
   action tensor.

Controlled comparison:

| condition | purpose |
|---|---|
| no guidance | base policy reference |
| `cumsum(action * 0.05)` ranking | old trajectory proxy baseline |
| forward-model ranking | proposed next method |
| current gradient guidance | existing guided implementation |

Use identical policy, environments, seeds, rollout counts, and collision
metrics.

Suggested first environment:

```text
PickPlaceBreadCerealCan
```

Then test harder setting:

```text
PickPlaceBreadCerealMilkCan
```

## Expected Outcomes

If ranking improves collision / clearance metrics:

```text
Continue toward point-cloud geometry input and replace oracle obstacle geometry.
```

If ranking does not improve metrics, but candidate analysis shows safe chunks
exist:

```text
The scoring / execution interface is still wrong; debug ranking cost and
closed-loop replanning.
```

If ranking shows the base policy rarely samples safe chunks:

```text
Inference-time geometry alone is insufficient. The project must introduce a
new training distribution, a learned critic, trajectory head, or multi-object
adaptation.
```

## What Not To Do Next

- Do not claim forward-model guidance works based on forward-model validation.
- Do not keep tuning gradient scale before checking whether safe candidate
  chunks exist.
- Do not revive Route B EEF-action training without a new learning formulation;
  expert Mink replay passing did not make learned EEF policy rollout work.
- Do not use the early 10-rollout scale sweep as final evidence; it is
  superseded by the larger controlled comparison.

## Reading Order For A Fresh Agent

1. This file.
2. `docs/route_b_validation/report.md`
3. `outputs/forward_model/osc_eef_forward_image_v15/summary.md`
4. `outputs/forward_model/random_rollout_validation/random_scale_0p3/SUMMARY.md`
5. `outputs/robomimic/eval/forward_model_controlled_comparison/SUMMARY.md`
6. `outputs/diagnostics/guidance_update_effect/SUMMARY.md`
7. `third_party/robomimic/AGENTS.md`

