# Checkpoint Retention

Status: active retention manifest, 2026-07-30.

Training outputs under `outputs/robomimic/train/` are local, ignored artifacts.
Keep only checkpoints that are required by a canonical result, comparison, or
the next bounded experiment. Training logs and configs remain in their run
directories.

## Retained checkpoints

| role | checkpoint |
|---|---|
| frozen delta-EEF source | `dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth` |
| LAN closure baseline | `lan_o3dp_can_delta_pose_eps_residual_40demo_seed42/20260724165723/models/model_epoch_350_image_v15_delta_eef_pose_osc_pointcloud_lan40_seed42_success_0.65.pth` |
| SDP 6k targeted-transfer checkpoint | `dp_can_delta_pose_osc_targeted_sdp/20260729173254/models/model_epoch_12.pth` |

All paths are relative to `outputs/robomimic/train/`.

## Cleanup record

On 2026-07-29, intermediate checkpoints and resumable `last*.pth` snapshots
from these completed runs were deleted after their result summaries and
canonical checkpoints were verified. The training directory decreased from
approximately 231 GiB to 18 GiB. Deleted checkpoints are not recoverable
except by rerunning the preserved configs and code.

On 2026-07-30, the recoverability boundary was archived. The SDP 10k,
positive-BC 50/50 and 90/10, clean-continuation, absolute-EEF, and DP3
comparison checkpoints were deleted. Their configs, logs, reports, and key
rollout results remain. Rejected 0.04-rad and duplicate C0 correction HDF5
files were also removed; the verified C1 raw and paired files are the retained
training data.

Do not delete:

- the correction raw / paired HDF5 data;
- fixed-state banks used by the support and branch diagnostics;
- any checkpoint in the table above;
- training configs, logs, result manifests, or canonical reports.

Before a future cleanup, update this manifest first and verify every retained
path exists.
