# Delta EEF to OSC MLP Adapter

Dataset: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5`
Model: `/home/yulei/codes/constraint-il-transfer/outputs/route_b_validation/delta_eef_to_osc_adapter/mlp_adapter.pth`
Device: `cuda`

## Offline Metrics

| split | mse_xyz | mae_xyz | cosine_median | clip_any_xyz |
|---|---:|---:|---:|---:|
| train | 0.000518872 | 0.0156129 | 0.9970 | 0.0000 |
| valid | 0.000784623 | 0.0188413 | 0.9960 | 0.0000 |

Offline metrics are not sufficient; this model still requires open-loop replay validation.
