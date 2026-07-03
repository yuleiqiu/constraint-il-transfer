# Delta EEF to OSC Adapter: Phase 2 Replay

Dataset: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5`
Split: `valid`
Demos: `demo_11, demo_126, demo_135, demo_140, demo_145`

## Replay Summary

| plan | desired cm | actual cm | tracking | mean target err cm | end orig err cm |
|---|---:|---:|---:|---:|---:|
| plan_A_original_osc | 1.419 | 0.354 | 0.253 | 0.407 | 0.346 |
| plan_B1_raw_delta_eef | 0.359 | 0.100 | 0.265 | 21.220 | 37.893 |
| adapter_scalar_scale | 1.339 | 0.349 | 0.259 | 3.494 | 8.840 |
| adapter_diagonal_scale | 1.339 | 0.350 | 0.259 | 3.254 | 8.848 |
| adapter_full_linear | 1.349 | 0.351 | 0.258 | 2.427 | 4.864 |
| adapter_mlp_state_conditioned | 1.410 | 0.359 | 0.257 | 2.301 | 7.002 |

## Interpretation Guide

- `plan_A_original_osc` is the replay upper bound: original dataset action through original OSC controller.
- `plan_B1_raw_delta_eef` is the known failure: actual EEF delta sent directly as an OSC command.
- Adapter plans transform `delta_eef_action[:3]` into OSC `actions[:3]`; rotation and gripper are copied from `delta_eef_action`.

Adapter plans tested:

- `scalar_scale`
- `diagonal_scale`
- `full_linear`
- `mlp_state_conditioned`

A usable adapter should be much closer to Plan A than to Plan B-1 in open-loop replay.
