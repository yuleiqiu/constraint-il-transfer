# Panda Mink Controller Outputs

This directory contains replay validation outputs for using robosuite's
Mink-based whole-body IK controller as a Panda EEF-native action interface.

## Directory Layout

| Path | Contents |
|---|---|
| `replay_valid5_default/` | Initial 5-demo smoke replay with default Panda Mink config. |
| `replay_valid50_default/` | Full validation split replay. The split has 20 demos, despite the requested `n=50`. |
| `replay_train50_default/` | First 50 train demos replay with default Panda Mink config. |
| `controller_configs/` | Full-pose controller configs with `ik_hand_ori_cost` set to `0.05`, `0.1`, and `0.5`. |
| `full_pose_valid_0p05/` | Full-pose replay on valid split with `ik_hand_ori_cost=0.05`. |
| `full_pose_valid_0p1/` | Full-pose replay on valid split with `ik_hand_ori_cost=0.1`. |
| `full_pose_valid_0p5/` | Full-pose replay on valid split with `ik_hand_ori_cost=0.5`. |
| `full_pose_train50_0p05/` | Full-pose replay on first 50 train demos with `ik_hand_ori_cost=0.05`. |
| `full_pose_train50_0p1/` | Full-pose replay on first 50 train demos with `ik_hand_ori_cost=0.1`. |
| `full_pose_train50_0p5/` | Full-pose replay on first 50 train demos with `ik_hand_ori_cost=0.5`. |
| `failure_videos_default/` | Mink replay videos for failed demos: `demo_140`, `demo_100`, `demo_101`, `demo_104`. |
| `failure_videos_osc_reference/` | Original OSC replay videos for the same failed demos, used as reference. |
| `STAGE_CONCLUSION.md` | Current stage conclusion and recommended next validation step. |

## Source Artifacts

Controller config:

```text
third_party/robosuite/robosuite/controllers/config/default/composite/panda_mink_ik.json
```

Replay script:

```text
docs/route_b_validation/verify_panda_mink_controller.py
```

Failure video script:

```text
docs/route_b_validation/render_panda_mink_failure_videos.py
```

Dataset used for replay:

```text
third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5
```

## Current Scope

The initial outputs validate position-only EEF tracking:

```text
action = [next_obs/robot0_eef_pos, 0, 0, 0, original_gripper_action]
ik_hand_ori_cost = 0.0
```

The full-pose validation uses:

```text
action = [next_obs/robot0_eef_pos, next_obs/robot0_eef_quat_site -> axis_angle, original_gripper_action]
ik_hand_ori_cost in {0.05, 0.1, 0.5}
```

`robot0_eef_quat_site` is used because it matches the controller target site
`gripper0_right_grip_site`; `robot0_eef_quat` is not the same orientation frame.
