# Delta EEF to OSC Adapter: Phase 1

Dataset: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5`

This phase only fits offline adapters. No environment replay was run.

## Best Offline Adapter

Best by valid `mse_xyz`: `full_linear`

## Valid Metrics

| adapter | mse_xyz | mae_xyz | cosine_median | clip_any_xyz | pred_norm_med | target_norm_med |
|---|---:|---:|---:|---:|---:|---:|
| scalar_scale | 0.00288155 | 0.0293788 | 0.9935 | 0.0003 | 0.2970 | 0.3159 |
| diagonal_scale | 0.00284533 | 0.0290102 | 0.9937 | 0.0003 | 0.2962 | 0.3159 |
| full_linear | 0.00251633 | 0.0292241 | 0.9924 | 0.0003 | 0.2977 | 0.3159 |

## Fitted Parameters

```json
{
  "scalar_scale": {
    "kind": "scalar_scale",
    "scale": 3.733987507238497
  },
  "diagonal_scale": {
    "kind": "diagonal_scale",
    "scale": [
      3.824432349700597,
      3.7892242357341543,
      3.5131057811356365
    ]
  },
  "full_linear": {
    "kind": "full_linear",
    "weights": [
      [
        3.637097825323677,
        0.2258082845612114,
        -0.3266486601043943
      ],
      [
        -0.05727897598130579,
        3.7465267551723214,
        -0.0013354680066856443
      ],
      [
        -0.18412419237478211,
        0.15951351039926892,
        3.4492034163634053
      ]
    ],
    "bias": [
      0.01608544095096622,
      0.0068380317094228074,
      0.015170357186739232
    ]
  }
}
```

## Interpretation

These metrics only test whether `delta_eef_action[:3]` predicts the dataset's original OSC `actions[:3]`.
The adapter is not validated for use until the next phase performs open-loop replay through robosuite.
