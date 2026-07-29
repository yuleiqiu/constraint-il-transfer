# Checkpoint Retention

Status: active retention manifest, 2026-07-29.

Training outputs under `outputs/robomimic/train/` are local, ignored artifacts.
Keep only checkpoints that are required by a canonical result, comparison, or
the next bounded experiment. Training logs and configs remain in their run
directories.

## Retained checkpoints

| role | checkpoint |
|---|---|
| frozen delta-EEF source | `dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth` |
| absolute-EEF comparison | `dp_can_abs_pose_osc/20260707222945/models/model_epoch_360_image_v15_abs_eef_pose_osc_success_0.82.pth` |
| DP3 point-cloud comparison | `dp3_can_delta_pose_osc_ddpm100_seed500/20260723192459/models/model_epoch_580_image_v15_delta_eef_pose_osc_pointcloud_success_0.82.pth` |
| LAN closure baseline | `lan_o3dp_can_delta_pose_eps_residual_40demo_seed42/20260724165723/models/model_epoch_350_image_v15_delta_eef_pose_osc_pointcloud_lan40_seed42_success_0.65.pth` |
| clean continuation endpoint | `dp_can_delta_pose_osc_clean_continuation_upper_bound/20260728205810/models/model_epoch_20.pth` |
| positive-BC 6k / 10k | `dp_can_delta_pose_osc_targeted_mixed_upper_bound/20260728204504/models/model_epoch_{12,20}.pth` |
| 90/10 BC 6k / 10k | `dp_can_delta_pose_osc_targeted_mixed_10pct_control/20260729171852/models/model_epoch_{12,20}.pth` |
| SDP 6k / 10k | `dp_can_delta_pose_osc_targeted_sdp/20260729173254/models/model_epoch_{12,20}.pth` |

All paths are relative to `outputs/robomimic/train/`.

## Cleanup record

On 2026-07-29, intermediate checkpoints and resumable `last*.pth` snapshots
from these completed runs were deleted after their result summaries and
canonical checkpoints were verified. The training directory decreased from
approximately 231 GiB to 18 GiB. Deleted checkpoints are not recoverable
except by rerunning the preserved configs and code.

Do not delete:

- the correction raw / paired HDF5 data;
- fixed-state banks used by the support and branch diagnostics;
- any checkpoint in the table above;
- training configs, logs, result manifests, or canonical reports.

Before a future cleanup, update this manifest first and verify every retained
path exists.
