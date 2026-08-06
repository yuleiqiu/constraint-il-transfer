# Paper-Aligned LAN-O3DP Reproduction

This experiment is a new reproduction path. It does not modify or overwrite
the retained delta-EEF LAN-O3DP artifacts from 2026-07-23.

The frozen observation and action contract is:

- absolute 7-D next-EEF action: world XYZ, site axis-angle, gripper;
- 6-D current EEF state: world XYZ and site axis-angle;
- one calibrated `agentview` camera; retained profiles render at 120x120 and
  the corrected high-density profile renders the same view at 256x256;
- visible metric-depth points selected with simulator-oracle geom masks;
- `paper256`: 256 target points (paper count), with deterministic repeat
  fallback only when fewer than 256 pixels are visible;
- `official512`: 512 target points (public simulation code count), with the
  public preprocessing code's zero-padding fallback.
- `paper256_hires_continuous`: corrected Robosuite transfer profile. It keeps
  the same calibrated `agentview` pose and world frame, renders oracle depth
  and segmentation at 256x256, and FPS-samples 256 unique target points. Its
  absolute EEF observations and actions use a quaternion-sign-continuous
  rotation-vector branch shared by offline generation and online rollout.

All generated datasets use the same fixed seed-42 subset and split: 40 total
demonstrations, 38 train, and validation demos `demo_18` and `demo_164`.
Each profile has 12,951 transitions. The corrected profile was generated and
verified locally on 2026-08-06: every frame has 256 unique points; the maximum
adjacent agent/action rotation-vector changes are 0.0422/0.0738 rad, with no
change above 0.1 rad.

Generate or resume the paper profile:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/create_dataset.py \
  --profile paper256
```

Generate the public-code point-count comparison:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/create_dataset.py \
  --profile official512
```

Generate the corrected high-density, continuous-rotation profile:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/create_dataset.py \
  --profile paper256_hires_continuous
```

This writes
`image_v15_abs_eef_pose_osc_lan40_paper256_hires_continuous.hdf5` and does
not modify either retained dataset above.

Verify structure plus offline/online observation parity:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/verify_dataset.py

MUJOCO_GL=egl uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/verify_dataset.py \
  --dataset third_party/robomimic/datasets/can/yq/\
image_v15_abs_eef_pose_osc_lan40_official512.hdf5

MUJOCO_GL=egl uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/verify_dataset.py \
  --dataset third_party/robomimic/datasets/can/yq/\
image_v15_abs_eef_pose_osc_lan40_paper256_hires_continuous.hdf5
```

## Training configurations

Training configurations are explicit rather than runtime mutations:

- `lan_o3dp_can_abs_pose_sample_paper256_seed42.json`: paper-primary profile
  and the paper's better-performing `sample` ablation;
- `lan_o3dp_can_abs_pose_epsilon_paper256_seed42.json`: epsilon-prediction
  base for the guidance equation used in the paper;
- `lan_o3dp_can_abs_pose_sample_official512_seed42.json`: public-code
  point-count comparison.
- `lan_o3dp_can_abs_pose_{sample,epsilon}_paper256_hires_continuous_seed42.json`:
  corrected 40-demo comparison runs.

All use DDPM-100, batch 64, 500 epochs, 2/16/8 observation/prediction/action
horizons, the residual 3-32-64-64 point encoder, the 512-1024-2048 UNet,
AdamW at `1e-4`, betas `(0.95, 0.999)`, weight decay `1e-6`, cosine decay with
300 warmup steps, and EMA power `0.75`. The formal runs, including both
corrected-profile configs, upload metrics to the W&B project
`lan_o3dp_paper_aligned`; TensorBoard and terminal logs are also retained
locally.

Run the paper-primary training:

```bash
ROBOMIMIC_GPU_ID=0 MUJOCO_GL=egl uv run python \
  third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/\
lan_o3dp_can_abs_pose_sample_paper256_seed42.json
```

The corrected sample-prediction config completed a real batch-64 preflight
with 251,759,200 parameters, finite loss, a save/reload/online-step round
trip, and 11.693 GiB peak allocated CUDA memory:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/preflight.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/\
lan_o3dp_can_abs_pose_sample_paper256_hires_continuous_seed42.json
```

## Paper-equation guidance

`paper_guided_lan_o3dp` is a new inference-only algorithm. It implements

```text
A_{k-1} <- scheduler_step(A_k) - rho * grad_{A_k} D(A_{0|k}, C_ob)
```

without detaching `A_{0|k}` from the denoiser. `C_ob` is chosen once per
action chunk as the visible obstacle point closest to the current EEF. The
thresholded XY cost is nonzero only inside `Q*`. The historical
`guided_lan_o3dp` implementation is unchanged.

The paired evaluator uses simulator-oracle masks from the same 120x120
`agentview`, metric depth, world coordinates, and visible FPS-128 obstacle
points. It reports Task SR, Safe SR, CR, NCR, the four-way outcome partition,
and paired failure-to-success / success-to-failure counts:

```bash
MUJOCO_GL=egl ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-05_lan_o3dp_paper_aligned/paired_eval.py \
  --agent /path/to/checkpoint.pth \
  --seeds 600 601 602 --n-rollouts 50 \
  --guidance-scale RHO --safety-distance Q_STAR \
  --out-dir outputs/lan_o3dp_paper_aligned/paired_eval
```

`rho` and `Q*` are deliberately required arguments: the paper treats them as
deployment parameters, and `Q*` depends on the grasped-object footprint. They
must be selected on a preregistered validation split before the held-out test
matrix.
