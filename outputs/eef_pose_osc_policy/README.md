# EEF Pose OSC Policy Findings

Date: 2026-07-08

## Conclusion

`delta_eef_pose_action` is an executable action target for clean-image diffusion
policy training and robosuite OSC control.

The learned policy's delta-EEF action chunk can also be converted back into the
EEF pose trajectory that the robot executes. This removes the need to use the
old OSC-action forward model for this action interface.

In this project, delta EEF pose is the replacement for the old forward-model
route: the policy output is already the EEF pose trajectory representation
needed for trajectory costs, ranking, or downstream guidance.

## Action Interface

The policy action is:

```text
delta_eef_pose_action = [
  next_obs/robot0_eef_pos - obs/robot0_eef_pos,
  axis_angle(R_next_site @ R_obs_site.T),
  original_gripper_action,
]
```

The controller is `OSC_POSE` with:

```text
input_type = delta
input_ref_frame = world
kp = 500
controller_goal_update_mode = desired
```

Quaternions use robosuite `xyzw` order directly.

## Trajectory Reconstruction

Given the current EEF pose `(p0, R0)` and an unnormalized policy action chunk
`a[0:H, 7]`, reconstruct the predicted pose trajectory as:

```text
p[t+1] = p[t] + a[t, 0:3]
R[t+1] = axisangle2mat(a[t, 3:6]) @ R[t]
```

The gripper action is executed as action dim 6 and is not part of the EEF pose
trajectory.

## Evidence

Diagnostic script:

```text
scripts/eef_pose_osc_policy/diagnose_delta_eef_policy_traj.py
```

Checkpoint:

```text
outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth
```

Command:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ROBOMIMIC_GPU_ID=0 \
uv run python scripts/eef_pose_osc_policy/diagnose_delta_eef_policy_traj.py \
  --n-rollouts 1 \
  --horizon 400 \
  --terminate-on-success \
  --output outputs/eef_pose_osc_policy/delta_policy_traj_diagnostic/epoch260_1rollout.json
```

Observed result:

```text
rollout success = 1/1
horizon = 334
chunks = 42
position error mean = 0.131 cm
position error p90 = 0.261 cm
position error max = 0.541 cm
orientation error mean = 0.253 deg
orientation error p90 = 0.563 deg
orientation error max = 1.019 deg
action_clip_count = 0
```

## Multi-Environment Evaluation

The epoch-260 checkpoint was evaluated for 50 episodes on each of three
evaluation seeds across all four PickPlace environments. Standard task success
decreased from 0.907 on PickPlaceCan to 0.200 on the three-distractor variant,
while non-target collision rate increased from 0.000 to 0.680.

The harder-environment failures are dominated by pre-target obstruction, but
the evaluation also contains collision-free placement failures and successful
episodes with incidental distractor contact. Future evaluation should report
Task SR together with Safe SR, CR, and NCR, and preserve the full four-way
partition: safe success, success with collision, collision failure, and NCR.

Full results and representative trajectories:
[`multienv_eval_report.md`](multienv_eval_report.md).

## Implication

For this action interface, guidance or action-chunk ranking can compute costs
directly on the policy-predicted EEF pose trajectory. The old learned forward
model is no longer part of the active code path; it is kept only as archived
evidence for why original OSC-action chunks were the wrong interface.
