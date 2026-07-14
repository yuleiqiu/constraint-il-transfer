# EEF-Pose OSC Policy Training

Run from the repository root with `uv run python ...`.

## Purpose

Train clean-image diffusion policies whose action target is executable EEF full-pose OSC, not the original robosuite OSC action.

Two action modes are supported:

- `abs_eef_pose_action`: `[next_eef_pos_world, axis_angle(next_eef_quat_site_xyzw), gripper]`
- `delta_eef_pose_action`: `[next_eef_pos_world - eef_pos_world, axis_angle(R_next_site @ R_site.T), gripper]`

Robosuite quaternions are xyzw; do not reorder them.

## Naming

Datasets keep the full interface name:

- `third_party/robomimic/datasets/can/yq/image_v15_abs_eef_pose_osc.hdf5`
- `third_party/robomimic/datasets/can/yq/image_v15_delta_eef_pose_osc.hdf5`

Training configs use shorter experiment names:

- abs: `dp_can_abs_pose_osc`
- delta: `dp_can_delta_pose_osc`

Both configs use the same W&B project name when W&B is enabled:

- `eef_pose_osc_policy`

## Generate Datasets

Delta:

```bash
uv run python scripts/eef_pose_osc_policy/create_delta_eef_pose_osc_dataset.py
```

Abs:

```bash
uv run python scripts/eef_pose_osc_policy/create_abs_eef_osc_dataset.py
```

Expected controller metadata:

- abs: `OSC_POSE`, `input_type=absolute`, `input_ref_frame=world`, `kp=500`
- delta: `OSC_POSE`, `input_type=delta`, `input_ref_frame=world`, `kp=500`, identity scaling, `controller_goal_update_mode=desired`

## Validate Before Training

Replay 10 demos first:

```bash
MUJOCO_GL=egl uv run python scripts/eef_pose_osc_policy/verify_delta_eef_pose_osc_dataset.py \
  --dataset third_party/robomimic/datasets/can/yq/image_v15_delta_eef_pose_osc.hdf5 \
  --n-demos 10

MUJOCO_GL=egl uv run python scripts/eef_pose_osc_policy/verify_abs_eef_osc_dataset.py \
  --dataset third_party/robomimic/datasets/can/yq/image_v15_abs_eef_pose_osc.hdf5 \
  --n-demos 10
```

Run wrapper smoke tests:

```bash
MUJOCO_GL=egl uv run python scripts/eef_pose_osc_policy/smoke_delta_eef_pose_osc_wrapper.py \
  --dataset third_party/robomimic/datasets/can/yq/image_v15_delta_eef_pose_osc.hdf5 \
  --config third_party/robomimic/robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_image.json

MUJOCO_GL=egl uv run python scripts/eef_pose_osc_policy/smoke_abs_eef_osc_wrapper.py \
  --dataset third_party/robomimic/datasets/can/yq/image_v15_abs_eef_pose_osc.hdf5 \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/diffusion_policy_can_image.json
```

These checks cover action dim, min-max action normalization, controller metadata, `reset_to` controller refresh, and whether raw actions are passed correctly to `env.step`.

## Delta Trajectory Diagnostic

After training a delta EEF policy, verify that policy action chunks reconstruct
the executed EEF pose trajectory:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ROBOMIMIC_GPU_ID=0 \
uv run python scripts/eef_pose_osc_policy/diagnose_delta_eef_policy_traj.py \
  --agent outputs/robomimic/train/dp_can_delta_pose_osc/<run>/models/<checkpoint>.pth \
  --n-rollouts 1 \
  --horizon 400 \
  --terminate-on-success \
  --output outputs/eef_pose_osc_policy/delta_policy_traj_diagnostic/<name>.json
```

Reference result: `outputs/eef_pose_osc_policy/README.md`.

## Multi-Env Diagnostic Eval

Before training more seeds, evaluate the current best delta EEF checkpoint over
the four PickPlace variants and three eval seeds:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ROBOMIMIC_GPU_ID=0 \
uv run python scripts/eef_pose_osc_policy/eval_delta_eef_multienv.py \
  --agent outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth \
  --envs PickPlaceCan PickPlaceBreadCan PickPlaceBreadCerealCan PickPlaceBreadCerealMilkCan \
  --seeds 600 601 602 \
  --n-rollouts 50 \
  --horizon 400 \
  --out-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50
```

The script writes:

```text
manifest.json
summary.csv
summary.md
<env>/seed_<seed>/episodes.hdf5
<env>/seed_<seed>/episode_metrics.jsonl
<env>/seed_<seed>/stats.json
```

It records raw and clipped actions, EEF pose trajectory, object poses and
displacements, contact counts, obstacle clearance, and per-chunk predicted vs.
executed EEF pose trajectories. MuJoCo states are optional via `--save-states`.

Analyze the eval output:

```bash
uv run python scripts/eef_pose_osc_policy/analyze_delta_eef_eval.py \
  --eval-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50
```

This writes `analysis/episode_features.csv`, `analysis/env_summary.csv`,
`analysis/analysis_summary.json`, and `analysis/analysis.md`.

Plot representative cases from the saved low-dimensional trajectories:

```bash
uv run python scripts/eef_pose_osc_policy/plot_delta_eef_eval_cases.py \
  --eval-dir outputs/eef_pose_osc_policy/eval/delta_epoch260_4env_3seed_n50 \
  --per-bucket 3
```

This writes `analysis/cases/case_index.md`, `analysis/cases/case_index.csv`,
and one PNG / JSON pair per selected case.

## Debug Train

Delta:

```bash
MUJOCO_GL=egl uv run python third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_image.json \
  --debug
```

Abs:

```bash
MUJOCO_GL=egl uv run python third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/diffusion_policy_can_image.json \
  --debug
```

`--debug` writes to `/tmp/tmp_trained_models` and should be cleaned after inspection.

## Full Train

Delta:

```bash
MUJOCO_GL=egl uv run python third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_image.json
```

Abs:

```bash
MUJOCO_GL=egl uv run python third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/diffusion_policy_can_image.json
```

Default configs use clean image observations, DDIM with `num_train_timesteps=100` and `num_inference_timesteps=10`, and min-max action normalization.

Training outputs go under:

- `outputs/robomimic/train/dp_can_delta_pose_osc/`
- `outputs/robomimic/train/dp_can_abs_pose_osc/`

Training-time rollout is serial in robomimic's train loop. If this is too slow, reduce `experiment.rollout.n` / increase `experiment.rollout.rate`, or disable training-time rollout and evaluate saved checkpoints with the parallel rollout script.

## Temporary Files

Safe cleanup after debug or parallel eval:

```bash
rm -rf /tmp/tmp_trained_models /tmp/parallel_worker_logs /tmp/worker_logs
rm -f /tmp/parallel_rollout_video_w*.mp4 /tmp/concat_list.txt
find /tmp -maxdepth 1 -type d -name 'pymp-*' -exec rm -rf {} +
find /tmp -maxdepth 2 -name '_remote_module_non_scriptable.py' -printf '%h\0' 2>/dev/null | xargs -0 -r rm -rf
rm -f /tmp/uv-*.lock /tmp/uv-setuptools-*.lock
```
