# LAN-O3DP Reproduction Phase 1: Point-Cloud Policy

Status: training chain validated; seed-500 production run is ready to launch,
2026-07-23.

This phase isolates the point-cloud policy training chain. It does not add
paper clean-action guidance, oracle obstacle rollout context, or multi-object
paired evaluation. Those begin only after the user confirms the final Phase-1
checkpoint.

## Fixed contract

- source demonstrations: clean `PickPlaceCan`;
- action: executable 7-D `delta_eef_pose_action`, min-max normalized;
- observation/action horizons: 2 / 16 / 8;
- diffusion: DDPM, epsilon prediction, 100 train and inference steps;
- policy seed: 500, at most 600 epochs;
- rollout selection: clean Task SR, then validation loss, then earlier epoch.

## Point-cloud observation

For every saved simulator state, `agentview` renders 256×256 normalized depth
and oracle per-geom segmentation. Depth is converted to metric camera depth,
and pixels belonging to physical `Can` or `VisualCan` goal geoms are
unprojected into world-frame XYZ. Transparent-goal pixels whose depth belongs
to an occluding or background geom are rejected by a 15 cm geom-center
consistency bound.

The remaining points are deterministically farthest-point sampled to 512.
When fewer than 512 valid points exist, the valid row-major sequence is
repeated cyclically. The same extractor is called by offline dataset generation
and the opt-in `EnvRobosuite` runtime provider.

The slim HDF5 keeps simulator states, the three low-dimensional policy
observations, `delta_eef_pose_action`, rewards/dones, train/valid masks,
per-demo metadata, and environment metadata. RGB and `next_obs` arrays are not
copied. Generation resumes at demo boundaries and rejects a mismatched
configuration hash unless overwrite is explicit.

## Encoder and training

`DP3PointCloudCore` consumes the processed scan layout `[3, 512]`. It applies
per-point linear layers `3→32→64→64`, LayerNorm, a residual 64-D block, and
max-pooling to a 64-D observation feature. Existing Diffusion Policy UNet and
EMA settings are unchanged.

Commands and Phase-1 verification checks are recorded in
`scripts/2026-07-23_lan_o3dp_reproduction/README.md`.

## Preflight result

The generated dataset contains all 200 demonstrations and 64,094 states and is
370 MB on disk. Its configuration hash is
`af5128d69d91303d85299b90f64063668ef4c02e44ef4018d4ca6647fb9cc3d3`.
Full HDF5 integrity and same-state offline/runtime point-cloud parity checks
pass. Re-running generation skips all completed demonstrations.

Point-cloud geometry and sampling tests, encoder shape / permutation /
padding / gradient tests, rollout checkpoint tie-breaking tests, and online
scan-layout processing tests pass. On one fixed full-dataset batch, loss fell
from 1.094030 to 0.256250 in 50 updates (76.6%). A two-epoch debug run
completed training, validation, checkpointing, and clean PickPlaceCan online
rollout with the point-cloud-only observation.
