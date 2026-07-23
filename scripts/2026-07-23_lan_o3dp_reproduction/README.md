# LAN-O3DP Reproduction: Phase 1

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
