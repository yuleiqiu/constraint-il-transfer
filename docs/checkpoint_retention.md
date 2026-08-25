# Checkpoint Retention

Status: active retention manifest, verified 2026-08-25.

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
| Empty2D failed mechanism-gate source | `empty2d_guidance_sample_seed42/20260807210551/models/model_epoch_100.pth` |
| Tabletop controlled-3D source | `lan_o3dp_tabletop_can_abs_pose_sample_100demo_seed42/20260814201328/models/model_epoch_100_tabletop_can_pointcloud_abs_eef_100_success_0.8.pth` |

All paths are relative to `outputs/robomimic/train/`.

## Closure verification

On 2026-08-25 all seven paths were present, SHA256-streamed, and loaded with
`torch.load(..., map_location="cpu")` as non-empty checkpoint dictionaries.

| role | bytes | SHA256 |
|---|---:|---|
| frozen delta-EEF source | 1,465,314,882 | `0a150c529b8d0d6dbf18b6bfd1a67417e7948536aed7f1d2499668e07d0357f8` |
| LAN closure baseline | 4,031,500,234 | `851305d48d34e2255d81bb103c1f0466eca293dc33261ade25225341736641d6` |
| SDP 6k targeted transfer | 1,465,254,854 | `166c000378c34891a29563990df23fdc4be0585239cafd848ba0b60bae3041c7` |
| paper-aligned sample-hires D2 | 4,028,750,482 | `921b16a9a00e3576204830a4745a876d99ad5a7ce611496bbf89f43bcebfe83c` |
| paper-aligned epsilon-hires | 4,028,751,380 | `e973ea67fbbe33163e42b492318ba481eac19b10cc56cf1fa9bc79fd826d1da6` |
| Empty2D epoch 100 | 1,044,552,106 | `fce1fd26a7b1f64d0cd1e5bdf7cd351dce96aca5163c0f52f53cad927eb5d1ba` |
| Tabletop epoch 100 | 4,028,678,950 | `e4c2993092122ecba7ae9dacc65e4b5d72caaec21f211fa2e4e92cd3de7cdcff` |

The SHA256 and byte size of the pre-Empty2D retained checkpoints are recorded
in `outputs/empty2d_guidance/checkpoint_cleanup_2026-08-07.json`.
The retained Empty2D epoch-100 checkpoint is 1,044,552,106 bytes with SHA256
`fce1fd26a7b1f64d0cd1e5bdf7cd351dce96aca5163c0f52f53cad927eb5d1ba`.
Its stage-close cleanup is recorded in
`outputs/empty2d_guidance/checkpoint_cleanup_2026-08-10.json`.

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

On 2026-08-10, the Empty2D stage was closed after its seed-42 mechanism gate
failed in both geometries. Epoch 100 is the only retained checkpoint because
it is the source used by the canonical paired result. Epochs 200/300/400/500
and the duplicate `last.pth` / `last_bak.pth` snapshots were deleted after the
selected checkpoint hash, merged result, reports, and training logs were
verified. Seeds 43/44 were not trained.

On 2026-08-18, the Tabletop controlled-3D guidance stage was closed as a
negative task-level result. The selected epoch-100 checkpoint was retained
with SHA256
`e4c2993092122ecba7ae9dacc65e4b5d72caaec21f211fa2e4e92cd3de7cdcff`;
epochs 50/150/200/250/300 and duplicate `last.pth` / `last_bak.pth` snapshots
were deleted, reclaiming 28,200,406,044 bytes. Training logs, W&B metadata,
formal source validation, fixed state banks, paired summaries, and canonical
videos remain. The exact audit is in
`outputs/tabletop_guidance/checkpoint_cleanup_candidates_2026-08-18.json`.

Also on 2026-08-18, every first-level output directory was reviewed. No other
retained checkpoint was deleted; superseded pilots, raw sweep partitions,
duplicate renders, and empty directories were removed. See
`outputs/CLEANUP_2026-08-18.md` for the all-stage audit.

Do not delete:

- the correction raw / paired HDF5 data;
- fixed-state banks used by the support and branch diagnostics;
- any checkpoint in the table above;
- training configs, logs, result manifests, or canonical reports.

Before a future cleanup, update this manifest first and verify every retained
path exists.
