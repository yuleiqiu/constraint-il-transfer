# OSC EEF Forward Model Reference

## Goal

Train a differentiable surrogate for the original OSC-command policy:

```text
f_hat(obs_t, osc_actions[t:t+H]) -> actual EEF trajectory[t+1:t+H]
```

This model is intended to replace the inaccurate guidance mapping:

```text
cumsum(action * 0.05)
```

It does not replace or retrain the diffusion policy.

## Dataset

Use the original OSC-command dataset:

```text
third_party/robomimic/datasets/can/yq/image_v15.hdf5
```

Fields used:

```text
actions
obs/robot0_eef_pos
obs/robot0_eef_quat
obs/robot0_gripper_qpos
next_obs/robot0_eef_pos
```

Fields intentionally not used:

```text
delta_eef_action
robot0_joint_pos
robot0_joint_vel
image observations
```

The input state is restricted to the low-dimensional observations already available to the current diffusion policy.

## Horizon

The horizon is not hard-coded. It is read from:

```text
third_party/robomimic/robomimic/exps/baseline/diffusion_policy_can_masked_image.json
algo.horizon.prediction_horizon
```

For the current DP config this resolves to `H=16`.

## Model

First version: residual MLP dynamics surrogate.

```text
state_t:
  robot0_eef_pos       3
  robot0_eef_quat      4
  robot0_gripper_qpos  2

action_chunk:
  actions[t:t+H]       H x 7

output:
  relative EEF trajectory, H x 3
```

The target is relative to the current EEF position:

```text
target[k] = next_obs/robot0_eef_pos[t+k] - obs/robot0_eef_pos[t]
```

## Baselines

Every evaluation compares against:

```text
hold baseline:
  predicted EEF position stays at obs/robot0_eef_pos[t]

cumsum baseline:
  obs/robot0_eef_pos[t] + cumsum(actions[t:t+H, :3] * 0.05)

train-fitted cumsum baselines:
  scalar / diagonal / full-linear one-step action-to-delta fits on train split,
  then accumulated over H steps
```

## Pass Criteria

The model should clearly beat the `cumsum(action * 0.05)` baseline.

Preferred target:

```text
valid trajectory RMSE < 1 cm
valid terminal error < cumsum terminal error by a large margin
```

Minimum useful result:

```text
valid trajectory / terminal errors are substantially below the previously observed 3-4 cm action-to-trajectory mapping error
```

If the forward model cannot beat the cumsum baseline, do not integrate it into diffusion guidance.

## Post-Validation Status

The model passed the offline held-out validation above and was integrated into
the obstacle-guidance rollout path. Later diagnostics narrowed the conclusion:

```text
forward model trajectory prediction: useful
current gradient-based obstacle guidance: not yet useful
```

In other words, the forward model should be treated as a trajectory predictor
or action-chunk evaluator. It should not be described as evidence that the
current guidance update improves rollout success.

## Planned Artifacts

This directory should contain:

```text
REFERENCE.md
config.json
model.pth
metrics.json
summary.md
```

`REFERENCE.md` is the pre-training plan. `summary.md` and `metrics.json` are generated after training.
