# Tabletop point-cloud guidance environment

This experiment isolates deployment-time avoidance in a controlled 3-D
pick-and-place task. The source demonstrations contain no obstacle. Evaluation
variants insert a static Milk or Cereal object into the direct transport
corridor while leaving it out of the policy observation.

## Contract

- environment: `PickPlaceCanTabletop`;
- robot / gripper: Panda / `RethinkGripper`;
- task: reach, grasp Can, lift, direct transport, lower into the shallow green
  tray, release, and retreat;
- source observation: 256 world-frame points from Can + GoalTray and the 6-D
  absolute EEF pose;
- action: next achieved absolute world EEF XYZ, site rotation vector, and
  gripper command;
- controller: `OSC_POSE`, absolute input, world frame, `kp=500`;
- source demos: no obstacle and no lateral avoidance waypoint.

The scripted teacher varies Can and tray XY, Can yaw, grasp offset, lift height,
phase speed, and open / close / release dwell. Each Cartesian segment uses a
minimum-jerk schedule. The carried-object transport segment remains direct.

## Teacher gate and data

The initial teacher gate passed 20/20 random layouts. The formal collector then
produced 100/100 successful demos (90 train / 10 validation), 25,599 total
transitions, and demo lengths 229--289. Start and goal samples cover their full
configured 6-by-6 cm regions. Maximum measured transport deviation from its
direct XY segment is 1.105 mm, caused by OSC tracking rather than a route
waypoint.

The formal dataset is:

```text
third_party/robomimic/datasets/can/yq/tabletop_can_pointcloud_abs_eef_100.hdf5
```

Reproduce collection and validation:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-14_tabletop_guidance/collect_demos.py \
  --output third_party/robomimic/datasets/can/yq/tabletop_can_pointcloud_abs_eef_100.hdf5 \
  --episodes 100 --seed 15000 --validation-fraction 0.1 --overwrite

MUJOCO_GL=egl uv run python \
  scripts/2026-08-14_tabletop_guidance/verify_demos.py \
  --dataset third_party/robomimic/datasets/can/yq/tabletop_can_pointcloud_abs_eef_100.hdf5
```

The HDF5 verifier checks all shapes and finite values, exact action alignment,
split disjointness, simulator replay, and one online controller step. Initial
replay errors are 0.00024 mm for the EEF vector and 0.033 mm for the unordered
point set. Consecutive action rotation-vector changes are at most 0.00434 rad,
so this fixed-orientation task does not cross the principal-axis discontinuity.

## DP baseline

Training uses the paper-aligned LAN-O3DP sample-prediction policy with DDPM-100,
2/16/8 observation/prediction/action horizons, the 256-point encoder, and the
512-1024-2048 1-D U-Net. Batch size is 256 on the 48-GiB RTX 6000 Ada. A real
CUDA update, checkpoint reload, policy action, and online environment step pass
at 13.863 GiB peak allocated memory.

Config:

```text
third_party/robomimic/robomimic/exps/absolute_eef_osc/
lan_o3dp_tabletop_can_abs_pose_sample_100demo_seed42.json
```

Formal training logs online to the W&B project
`tabletop_guidance_baseline` and saves checkpoints every 50 epochs:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
ROBOMIMIC_GPU_ID=0 WANDB_MODE=online uv run python \
  third_party/robomimic/robomimic/scripts/train.py \
  --config third_party/robomimic/robomimic/exps/absolute_eef_osc/\
lan_o3dp_tabletop_can_abs_pose_sample_100demo_seed42.json
```

The rollout screen selected epoch 100. The clean-task success rates measured
by the 10-rollout training screen at epochs 50 / 100 / 150 / 200 / 250 / 300
were 0.70 / 0.80 / 0.40 / 0.50 / 0.60 / 0.50. Training was therefore stopped
after epoch 301 instead of spending the remaining compute on checkpoints whose
closed-loop performance had already degraded, despite their lower diffusion
loss.

Selected checkpoint:

```text
outputs/robomimic/train/
lan_o3dp_tabletop_can_abs_pose_sample_100demo_seed42/20260814201328/models/
model_epoch_100_tabletop_can_pointcloud_abs_eef_100_success_0.8.pth
```

An independent 50-rollout clean evaluation with diffusion seeds 26000--26049
and horizon 600 obtains 35/50 task successes (`Task SR = 0.70`, Wilson 95%
interval 0.562--0.809). Successful rollouts take 205--534 steps, with a mean
of 367.8; two successes occur after step 500, which is why the formal horizon
is 600. The 15 failures comprise four failed grasps, six post-lift stalls near
the source, two stalls over the goal, and three completed placements without
enough retreat to satisfy the inherited strict PickPlaceCan success condition.

The complete per-rollout record is:

```text
outputs/tabletop_guidance/baseline_epoch100_eval50_h600_seed26000.json
```

The online W&B run is in project `tabletop_guidance_baseline` (run
`58wfbbrn`). This is a valid executable closed-loop DP baseline, but its 0.70
clean Task SR is a reliability limitation that must be carried into later
obstacle / guidance comparisons.
