# Stage Conclusion: Panda Mink EEF Controller

Date: 2026-06-29

## Question

Can robosuite's Mink-based whole-body IK controller provide a usable
EEF-native action interface for Route B?

## Setup

Controller:

```text
WHOLE_BODY_MINK_IK
ref_name = gripper0_right_grip_site
ik_input_type = absolute
ik_input_ref_frame = world
ik_input_rotation_repr = axis_angle
ik_hand_pos_cost = 1.0
ik_hand_ori_cost = 0.0
```

Action used in replay:

```text
[next_obs/robot0_eef_pos, 0, 0, 0, original_gripper_action]
```

This is a position-only EEF validation. Orientation tracking was intentionally
disabled for this stage.

## Replay Results

| Run | Demos | Success final | Tracking ratio | Mean target error | End original error |
|---|---:|---:|---:|---:|---:|
| `replay_valid5_default` | 5 | not logged | 0.215 | 1.272 cm | 0.271 cm |
| `replay_valid50_default` | 20 | 19/20 = 95% | 0.215 | 1.324 cm | 0.267 cm |
| `replay_train50_default` | 50 | 47/50 = 94% | 0.215 | 1.353 cm | 0.264 cm |

Failed demos:

```text
valid: demo_140
train: demo_100, demo_101, demo_104
```

## OSC Reference Check

The four Mink-failed demos were replayed with the original OSC actions.

| Mode | Demos | Success final | Max EEF error |
|---|---:|---:|---:|
| Original OSC | 4 | 4/4 = 100% | 0.06-0.08 cm |
| Panda Mink position-only | 4 | 0/4 = 0% | 2.1-6.1 cm |

This shows that the failed cases are not caused by broken dataset states. The
failure is introduced by the new EEF controller interface.

## Interpretation

The Panda Mink controller is usable as a first EEF-native controller prototype:

```text
valid success: 95%
train50 success: 94%
final EEF drift: about 0.26 cm
```

However, it is not a strict drop-in replacement for the original OSC interface.
The controller has a clear response lag:

```text
actual per-step EEF motion / requested per-step EEF motion ~= 0.215
```

The likely failure mode is contact / grasp timing sensitivity caused by this
lag, not global trajectory drift.

## Full-Pose Validation

Full-pose replay was run after the position-only validation. The orientation
target is:

```text
next_obs/robot0_eef_quat_site -> axis_angle
```

`robot0_eef_quat_site` was selected because it matches the MuJoCo orientation
of `gripper0_right_grip_site`, which is the Mink controller's `ref_name`.
`robot0_eef_quat` does not match that site frame.

Full-pose action:

```text
[next_obs/robot0_eef_pos, next_obs/robot0_eef_quat_site -> axis_angle, original_gripper_action]
```

Full-pose results:

| Run | Orientation cost | Demos | Success final | Mean ori target error | Mean target pos error | End original pos error |
|---|---:|---:|---:|---:|---:|---:|
| `full_pose_valid_0p05` | 0.05 | 20 | 20/20 = 100% | 7.86 deg | 1.312 cm | 0.267 cm |
| `full_pose_valid_0p1` | 0.1 | 20 | 20/20 = 100% | 2.63 deg | 1.319 cm | 0.279 cm |
| `full_pose_valid_0p5` | 0.5 | 20 | 20/20 = 100% | 0.89 deg | 1.330 cm | 0.371 cm |
| `full_pose_train50_0p05` | 0.05 | 50 | 50/50 = 100% | 7.93 deg | 1.327 cm | 0.259 cm |
| `full_pose_train50_0p1` | 0.1 | 50 | 49/50 = 98% | 2.76 deg | 1.336 cm | 0.266 cm |
| `full_pose_train50_0p5` | 0.5 | 50 | 49/50 = 98% | 0.96 deg | 1.362 cm | 0.322 cm |

Failed full-pose demos:

```text
ori_cost=0.05: none
ori_cost=0.1:  demo_104
ori_cost=0.5:  demo_104
```

## Controller Decision

For controller validation, full-pose EEF replay is the best-tested interface.

Recommended controller setting:

```text
ik_hand_pos_cost = 1.0
ik_hand_ori_cost = 0.05
orientation source = robot0_eef_quat_site
```

Rationale:

```text
position-only valid/train50 success: 95% / 94%
full-pose ori_cost=0.05 valid/train50 success: 100% / 100%
```

Higher orientation cost improves orientation error but begins to hurt task
success and positional replay quality. For Route B, task success and contact
timing are more important than sub-degree orientation tracking, so `0.05` is
the pragmatic setting.

## Training Follow-Up

The full-pose controller result did not transfer to learned-policy rollout.

A full-pose EEF dataset and training run were created:

```text
third_party/robomimic/datasets/can/yq/image_v15_abs_eef_pose_mink.hdf5
outputs/robomimic/train/diffusion_policy_can_yq_abs_eef_pose_mink_image/20260629183845
```

The rollout result was:

```text
Epoch 20: Success_Rate = 0.0
```

This updates the stage conclusion:

```text
expert EEF replay through Panda Mink IK: passes
learned EEF target policy through Panda Mink IK: fails
```

Therefore Panda Mink IK is useful as an EEF-native replay diagnostic, but it is
not currently a working Route B training solution. The remaining failure is
closed-loop policy learning: replay follows expert absolute EEF targets and
expert gripper timing, while rollout requires the learned policy to generate
EEF targets from its own visited states.

## Related Training Smoke Test

A position-only absolute EEF dataset and config were created and passed a
robomimic `--debug` smoke test:

```text
third_party/robomimic/datasets/can/yq/image_v15_abs_eef_mink.hdf5
third_party/robomimic/robomimic/exps/delta_eef/diffusion_policy_can_abs_eef_mink_image.json
```

The smoke test confirmed that training, validation, W&B logging, checkpoint
saving, rollout video writing, and Panda Mink rollout env creation all work.

This smoke test has been superseded by the full-pose dataset and the failed
full-pose policy training run above.
