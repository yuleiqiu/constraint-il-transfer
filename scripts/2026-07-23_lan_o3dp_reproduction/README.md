# LAN-O3DP Reproduction

This directory builds and verifies the point-cloud-conditioned delta-EEF
Diffusion Policy training chain. It does not implement paper guidance or
multi-object evaluation.

Generate or resume the full dataset:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-23_lan_o3dp_reproduction/create_target_pointcloud_dataset.py
```

An existing output must have the same configuration hash. Use `--overwrite`
only to explicitly regenerate it.

Run Phase-1 checks:

```bash
uv run python third_party/robomimic/tests/test_target_pointcloud_utils.py
uv run python third_party/robomimic/tests/test_dp3_pointcloud_core.py
uv run python third_party/robomimic/tests/test_rollout_checkpoint_selection.py
uv run python third_party/robomimic/tests/test_rollout_scan_processing.py

MUJOCO_GL=egl uv run python \
  scripts/2026-07-23_lan_o3dp_reproduction/verify_phase1_dataset.py

uv run python \
  scripts/2026-07-23_lan_o3dp_reproduction/overfit_one_batch.py

MUJOCO_GL=egl uv run python \
  third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_pointcloud_ddpm100.json \
  --debug
```

The production config uses DDPM epsilon prediction with 100 train and
inference steps, horizons 2/16/8, seed 500, and at most 600 epochs. Training
rollouts use only clean `PickPlaceCan`. Checkpoints are ranked by Task SR, then
validation loss, then earlier epoch.

## LAN-aligned 40-demo run

Create the fixed seed-42 subset without modifying either source HDF5:

```bash
uv run python \
  scripts/2026-07-23_lan_o3dp_reproduction/create_lan40_subset.py
```

This selects 40 IDs from the numeric ordering of all 200 demos, then uses a
second independent seed-42 generator to hold out `demo_18` and `demo_164`.
The adjacent manifest records both source SHA-256 fingerprints, the 38/2
split, initial Can XY/yaw distribution audit, and per-trajectory copy hashes.

Run the fixed batch-64 memory, checkpoint reload, and one-step environment
gate:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-07-23_lan_o3dp_reproduction/preflight_lan40.py
```

Run the standard debug train and the one production train:

```bash
MUJOCO_GL=egl uv run python \
  third_party/robomimic/robomimic/scripts/train.py \
  --config \
  third_party/robomimic/robomimic/exps/delta_eef_pose_osc/lan_o3dp_can_delta_pose_eps_residual_40demo_seed42.json \
  --debug

MUJOCO_GL=egl uv run python \
  third_party/robomimic/robomimic/scripts/train.py \
  --config \
  third_party/robomimic/robomimic/exps/delta_eef_pose_osc/lan_o3dp_can_delta_pose_eps_residual_40demo_seed42.json
```

The separately registered `lan_o3dp` path uses the LAN point encoder,
disables robomimic's extra post-encoder ReLU, and consumes the configured
512/1024/2048 UNet. Observation min-max ranges come only from the 38 training
demos. The deliberate project deviations are:

- executable 7-D delta EEF pose actions instead of absolute EEF actions;
- 9-D EEF position, quaternion, and gripper state instead of RLBench's 6-D
  `agent_pos`;
- 512 merged Can + VisualCan points from one front camera, matching the public
  execution chain; and
- epsilon prediction, retained for later guided denoising.

No guided variant or additional control training is part of this run.
