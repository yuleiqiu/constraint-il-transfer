# Tabletop wide-support source policy

Status: completed; no reliable task-level guidance gain. Read
`outputs/tabletop_wide_support/conclusion.md` from project root before using
these reproduction commands. This experiment uses original XY nearest-point
guidance, not the MVEE implementation present on the same branch.

This experiment transfers the Empty2D data principle to the controlled 3-D
Tabletop task. Demonstrations are collected in obstacle-free
`PickPlaceCanTabletop`; Milk is absent and never enters the teacher or policy
observation. The carried-Can transport segment is a straight-line mean plus a
symmetric endpoint-conditioned RBF-GP lateral residual. Post-hoc gates measure
near-direct, upper, and lower route support.

The purpose is to test whether predicted-clean guidance can select a safe
task-completing route from broad source-policy support. It does not train an
obstacle-conditioned avoidance policy.

Run the pure generator checks:

```bash
uv run pytest -q scripts/2026-08-26_tabletop_wide_support/test_wide_support_teacher.py
```

Run the fixed 20-layout closed-loop teacher gate:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-26_tabletop_wide_support/evaluate_teacher.py \
  --episodes 20 --seed 16000 \
  --out outputs/tabletop_wide_support/teacher_gate_seed16000_n20.json
```

Collect and validate the fixed 100-demo dataset:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-26_tabletop_wide_support/collect_demos.py \
  --output third_party/robomimic/datasets/can/yq/tabletop_can_wide_support_pointcloud_abs_eef_100.hdf5 \
  --episodes 100 --seed 17000 --validation-fraction 0.1

MUJOCO_GL=egl uv run python \
  scripts/2026-08-26_tabletop_wide_support/verify_demos.py \
  --dataset third_party/robomimic/datasets/can/yq/tabletop_can_wide_support_pointcloud_abs_eef_100.hdf5
```

Visualize achieved TCP transport coverage before training:

```bash
uv run python \
  scripts/2026-08-26_tabletop_wide_support/visualize_tcp_coverage.py \
  --dataset third_party/robomimic/datasets/can/yq/tabletop_can_wide_support_pointcloud_abs_eef_100.hdf5 \
  --output outputs/tabletop_wide_support/tcp_transport_coverage.png \
  --summary outputs/tabletop_wide_support/tcp_transport_coverage.json
```

Train the architecture-matched LAN-O3DP source:

```bash
MUJOCO_GL=egl ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-26_tabletop_wide_support/preflight_training.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/\
lan_o3dp_tabletop_can_wide_support_abs_pose_sample_100demo_seed42.json

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
ROBOMIMIC_GPU_ID=0 WANDB_MODE=online uv run python \
  third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/\
lan_o3dp_tabletop_can_wide_support_abs_pose_sample_100demo_seed42.json
```

Screen source checkpoints on an exact shared clean-state bank:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-12_lan_o3dp_hand_off_guidance/create_state_bank.py \
  --agent <checkpoint> --envs PickPlaceCanTabletop \
  --seed 827 --n-rollouts 50 \
  --out-dir outputs/tabletop_wide_support/fixed_clean_state_bank_seed827_n50

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
ROBOMIMIC_GPU_ID=0 uv run python \
  scripts/2026-08-26_tabletop_wide_support/fixed_state_checkpoint_screen.py \
  --state-bank outputs/tabletop_wide_support/fixed_clean_state_bank_seed827_n50/initial_state_bank.pkl \
  --bank-seed 827 --episodes 50 --horizon 600 \
  --checkpoint epoch50=<epoch-50-checkpoint> \
  --checkpoint epoch250=<epoch-250-checkpoint> \
  --out-dir outputs/tabletop_wide_support/fixed_clean_checkpoint_screen_seed827_n50
```

The two checkpoints can also be evaluated concurrently by invoking the screen
once per checkpoint with the same state bank and bank seed, while assigning
the processes to different GPUs. Pair the resulting JSONL rows by episode,
initial-state hash, and policy seed.

```bash
uv run python \
  scripts/2026-08-26_tabletop_wide_support/aggregate_fixed_checkpoint_screen.py \
  --left <epoch50.jsonl> --right <epoch250.jsonl> --out <paired_summary.json>
```

Evaluate baseline and the fixed 3-D Huber-hinge guidance condition on a shared
RaisedMilk state bank with `raised_milk_condition_eval.py`, assigning one
condition to each GPU. Merge the two JSONL files with
`aggregate_raised_milk_eval.py`; the merger reports Task SR, Safe SR, CR, NCR,
the four-way outcome partition, and paired task / safe transitions.

For the transport-support hypothesis, pass `--guidance-phase target_lifted` so
the 3-D intervention begins only after the Can is at least 8 cm above the
table. This is the phase corresponding to Empty2D's single-stage path motion;
the archived `always` condition also modifies reach and grasp.

Use `render_target_lifted_pairs.py` to replay selected paired outcomes from the
same saved state and policy seed and write synchronized baseline-versus-guided
videos. The video manifest records whether task and collision outcomes replay
the formal evaluation.
