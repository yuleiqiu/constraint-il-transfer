# LAN-O3DP Phase 1 Training Report

Status: training complete; clean-source acceptance gate failed, 2026-07-23.

## Run

- run directory:
  `outputs/robomimic/train/dp3_can_delta_pose_osc_ddpm100_seed500/20260723192459`;
- W&B run: `07kqvfgm`;
- seed: 500;
- schedule: 600 epochs, 500 train batches and 50 validation batches per
  epoch;
- evaluation: 50 clean `PickPlaceCan` rollouts every 20 epochs, horizon 400,
  terminate on success;
- checkpoint ranking: Task SR, then validation loss, then earlier epoch.

The run completed normally and W&B synced successfully. It evaluated 30
periodic candidates over 1,500 clean rollouts.

## Acceptance outcome

The best candidate is epoch 580:

`outputs/robomimic/train/dp3_can_delta_pose_osc_ddpm100_seed500/20260723192459/models/model_epoch_580_image_v15_delta_eef_pose_osc_pointcloud_success_0.82.pth`

It obtains Task SR 0.82 (41 / 50) with validation loss 0.304707. The Phase-1
gate requires Task SR at least 0.90 (45 / 50), so the run falls four successful
episodes short. The final epoch-600 candidate obtains Task SR 0.64. Reloading
the best checkpoint recreates the point-cloud rollout environment and produces
a finite 7-D action from a `[2, 512, 3]` frame-stacked observation.

Decision: Phase 1 does not pass. Do not start paper guidance, multi-object
paired rollout, or Phase 2 checkpoint adaptation. The epoch-580 checkpoint is
the empirical best candidate, not an accepted final checkpoint.

## All candidates

| Epoch | Task SR | Validation loss | Checkpoint |
|---:|---:|---:|---|
| 20 | 0.06 | 0.052605 | `model_epoch_20_image_v15_delta_eef_pose_osc_pointcloud_success_0.06.pth` |
| 40 | 0.24 | 0.048225 | `model_epoch_40_image_v15_delta_eef_pose_osc_pointcloud_success_0.24.pth` |
| 60 | 0.46 | 0.058748 | `model_epoch_60_image_v15_delta_eef_pose_osc_pointcloud_success_0.46.pth` |
| 80 | 0.48 | 0.075498 | `model_epoch_80_image_v15_delta_eef_pose_osc_pointcloud_success_0.48.pth` |
| 100 | 0.66 | 0.085973 | `model_epoch_100_image_v15_delta_eef_pose_osc_pointcloud_success_0.66.pth` |
| 120 | 0.66 | 0.108505 | `model_epoch_120.pth` |
| 140 | 0.66 | 0.124780 | `model_epoch_140.pth` |
| 160 | 0.60 | 0.120032 | `model_epoch_160.pth` |
| 180 | 0.62 | 0.144111 | `model_epoch_180.pth` |
| 200 | 0.66 | 0.156646 | `model_epoch_200.pth` |
| 220 | 0.78 | 0.172537 | `model_epoch_220_image_v15_delta_eef_pose_osc_pointcloud_success_0.78.pth` |
| 240 | 0.72 | 0.175192 | `model_epoch_240.pth` |
| 260 | 0.80 | 0.164244 | `model_epoch_260_image_v15_delta_eef_pose_osc_pointcloud_success_0.8.pth` |
| 280 | 0.64 | 0.186462 | `model_epoch_280.pth` |
| 300 | 0.66 | 0.205992 | `model_epoch_300.pth` |
| 320 | 0.70 | 0.225048 | `model_epoch_320.pth` |
| 340 | 0.78 | 0.230299 | `model_epoch_340.pth` |
| 360 | 0.74 | 0.233829 | `model_epoch_360.pth` |
| 380 | 0.78 | 0.237251 | `model_epoch_380.pth` |
| 400 | 0.64 | 0.259129 | `model_epoch_400.pth` |
| 420 | 0.62 | 0.297255 | `model_epoch_420.pth` |
| 440 | 0.74 | 0.290451 | `model_epoch_440.pth` |
| 460 | 0.68 | 0.277184 | `model_epoch_460.pth` |
| 480 | 0.70 | 0.310234 | `model_epoch_480.pth` |
| 500 | 0.76 | 0.318666 | `model_epoch_500.pth` |
| 520 | 0.74 | 0.316380 | `model_epoch_520.pth` |
| 540 | 0.78 | 0.327588 | `model_epoch_540.pth` |
| 560 | 0.78 | 0.284018 | `model_epoch_560.pth` |
| 580 | **0.82** | 0.304707 | `model_epoch_580_image_v15_delta_eef_pose_osc_pointcloud_success_0.82.pth` |
| 600 | 0.64 | 0.329323 | `model_epoch_600.pth` |
