# Checkpoint Retention

Status: active retention manifest, 2026-08-07.

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
| paper-aligned sample-hires D2 checkpoint | `lan_o3dp_can_abs_pose_sample_paper256_hires_continuous_seed42/20260806121402/models/model_epoch_50_image_v15_abs_eef_pose_osc_lan40_paper256_hires_continuous_success_0.7.pth` |
| paper-aligned epsilon-hires comparison | `lan_o3dp_can_abs_pose_epsilon_paper256_hires_continuous_seed42/20260806121403/models/model_epoch_300_image_v15_abs_eef_pose_osc_lan40_paper256_hires_continuous_success_0.5.pth` |

All paths are relative to `outputs/robomimic/train/`.

The SHA256 and byte size of every retained checkpoint are recorded in
`outputs/empty2d_guidance/checkpoint_cleanup_2026-08-07.json`.

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

On 2026-07-31, the repeated-terminal Toy source was superseded by an
absorbing-terminal retraining. Its five checkpoints, latent-dynamics model,
and rollout HDF5 were deleted after compact audit artifacts were moved under
`outputs/toy_recovery_objectives/archive/`. Two pre-delta-EEF image-action
checkpoints under `outputs/robomimic/checkpoints/` were also deleted. Completed
evaluation `episodes.hdf5` payloads and source/absolute-EEF training videos
were removed after their summaries and manifests were verified. The outputs
tree decreased from approximately 14 GiB to 8.4 GiB. These deleted local
artifacts are recoverable only by rerunning preserved code and configs or by
checking out their historical implementation state.

On 2026-08-07, the paper-aligned LAN stage was archived. The two low-resolution
sample/epsilon run directories and all non-canonical checkpoint snapshots in
the two high-resolution continuous-rotation runs were deleted. The sample
epoch-50 checkpoint used by the D2 report and the epsilon epoch-300 comparison
were retained. The superseded `toy_recovery` package, dated scripts, complete
`outputs/toy_recovery_objectives/` tree, and its five absorbing-source
checkpoints were deleted as explicitly planned; tracked content remains
recoverable from Git history. See the cleanup manifest above for the exact
retained hashes and before/after disk audit.

Do not delete:

- the correction raw / paired HDF5 data;
- fixed-state banks used by the support and branch diagnostics;
- any checkpoint in the table above;
- training configs, logs, result manifests, or canonical reports.

Before a future cleanup, update this manifest first and verify every retained
path exists.
