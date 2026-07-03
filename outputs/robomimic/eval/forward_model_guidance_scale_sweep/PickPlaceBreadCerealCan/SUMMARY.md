# Forward-Model Guidance Scale Sweep

Environment: `PickPlaceBreadCerealCan`

## Status

This was an early 10-rollout scale sweep. Its optimistic candidate-setting
conclusion is superseded by the later 3-seed x 50-rollout controlled
comparison:

```text
outputs/robomimic/eval/forward_model_controlled_comparison/SUMMARY.md
```

The controlled comparison found that both cumsum guidance and forward-model
guidance reduced success and increased collision metrics relative to no
guidance. Treat this file only as a small exploratory sweep, not as evidence
that forward-model guidance works.

Common settings:
- checkpoint: `outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth`
- trajectory backend: `forward_model`
- forward model: `outputs/forward_model/osc_eef_forward_image_v15/model.pth`
- guidance geometry: `oracle_center`
- guidance mode: `xy`
- guidance horizon: `8`
- rollouts: `10`
- workers: `4`
- seed: `700`
- video: enabled, `video_skip=5`

Baseline reference from previous smoke run:

| method | success | collision_any | collision_steps | collision_rate |
|---|---:|---:|---:|---:|
| no_guidance | 9/10 | 0.40 | 31.0 | 0.0798 |
| forward_model_guidance, scale=0.03 | 9/10 | 0.40 | 21.3 | 0.0557 |

Scale sweep results:

| scale | success | collision_any | collision_steps | collision_rate | positive_cost_rate | guidance_cost |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 10/10 | 0.30 | 13.8 | 0.0403 | 0.0975 | 0.00023146 |
| 0.01 | 9/10 | 0.40 | 7.2 | 0.0213 | 0.0796 | 0.00019831 |
| 0.02 | 9/10 | 0.30 | 16.6 | 0.0448 | 0.0935 | 0.00018053 |
| 0.03 | 10/10 | 0.40 | 12.6 | 0.0340 | 0.0720 | 0.00014432 |

Videos:

| scale | rollout video | failure clip |
|---:|---|---|
| 0.005 | `scale_0.005/seed_700/rollouts.mp4` | none |
| 0.01 | `scale_0.01/seed_700/rollouts.mp4` | `scale_0.01/seed_700/failure_rollout_6.mp4` |
| 0.02 | `scale_0.02/seed_700/rollouts.mp4` | `scale_0.02/seed_700/failure_rollout_6.mp4` |
| 0.03 | `scale_0.03/seed_700/rollouts.mp4` | none |

Failure mapping:
- `n_rollouts=10`, `n_workers=4`
- worker assignment: `worker_0: [0,1,2]`, `worker_1: [3,4,5]`, `worker_2: [6,7]`, `worker_3: [8,9]`
- the only task failures are rollout `6` for scales `0.01` and `0.02`
- the failure clips are the first 4 seconds of `worker_2.mp4`, corresponding to rollout `6` with horizon `400`

Interpretation:
- In this single seed, forward-model guidance reduced collision duration for
  some scales.
- The effect is not monotonic in scale and does not reliably remove collision occurrence.
- `scale=0.005` looked like the best candidate in this sweep, but the larger
  controlled comparison did not validate it.
