# Forward Model Controlled Comparison

Date: 2026-07-02

Question:

Does replacing the old `cumsum(action * 0.05)` trajectory approximation with the learned OSC forward model improve obstacle-guided rollout performance?

## Setup

- Policy: `outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth`
- Environment: `PickPlaceBreadCerealCan`
- Rollouts: `3 seeds x 50 = 150` per condition
- Seeds: `700, 701, 702`
- Workers: `10`
- Obstacle source: `oracle_center`
- Guidance mode: `xy`
- Guidance horizon: `8`
- Guidance scale: `0.005`
- Video: disabled

Conditions:

1. `no_guidance`
2. `cumsum_guidance_scale_0.005`
3. `forward_model_guidance_scale_0.005`

## Per-Seed Results

| group | seed | success | collision_any | collision_steps | collision_rate | positive_count | positive_rate | guidance_cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_guidance | 700 | 46/50 | 0.26 | 11.94 | 0.0321 | 0.00 | 0.0000 | 0.00000000 |
| no_guidance | 701 | 45/50 | 0.26 | 15.80 | 0.0411 | 0.00 | 0.0000 | 0.00000000 |
| no_guidance | 702 | 46/50 | 0.26 | 13.62 | 0.0366 | 0.00 | 0.0000 | 0.00000000 |
| cumsum_guidance_scale_0.005 | 700 | 45/50 | 0.28 | 15.04 | 0.0399 | 4.06 | 0.0986 | 0.00013334 |
| cumsum_guidance_scale_0.005 | 701 | 44/50 | 0.28 | 15.42 | 0.0404 | 3.76 | 0.0916 | 0.00011316 |
| cumsum_guidance_scale_0.005 | 702 | 42/50 | 0.30 | 21.44 | 0.0557 | 3.74 | 0.0916 | 0.00010853 |
| forward_model_guidance_scale_0.005 | 700 | 45/50 | 0.24 | 17.40 | 0.0455 | 3.56 | 0.0867 | 0.00028617 |
| forward_model_guidance_scale_0.005 | 701 | 42/50 | 0.30 | 16.54 | 0.0431 | 3.38 | 0.0790 | 0.00031025 |
| forward_model_guidance_scale_0.005 | 702 | 45/50 | 0.32 | 14.86 | 0.0395 | 3.32 | 0.0781 | 0.00032887 |

## Aggregate Results

| group | success | collision_any | collision_rollouts | collision_steps | collision_rate | positive_count | positive_rate | guidance_cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_guidance | 137/150 = 0.913 | 0.260 | 39/150 | 13.79 | 0.0366 | 0.00 | 0.0000 | 0.00000000 |
| cumsum_guidance_scale_0.005 | 131/150 = 0.873 | 0.287 | 43/150 | 17.30 | 0.0453 | 3.85 | 0.0939 | 0.00011834 |
| forward_model_guidance_scale_0.005 | 132/150 = 0.880 | 0.287 | 43/150 | 16.27 | 0.0427 | 3.42 | 0.0813 | 0.00030843 |

## Conclusion

This controlled rollout comparison does not support the claim that forward-model guidance is useful at `scale=0.005`.

Key points:

- Both guided conditions reduce success compared with no guidance.
- Both guided conditions increase collision-any compared with no guidance.
- Forward-model guidance is slightly better than cumsum on collision duration:
  - cumsum collision steps: `17.30`
  - forward-model collision steps: `16.27`
- But forward-model guidance is still worse than no guidance:
  - no-guidance collision steps: `13.79`
- Therefore, the learned forward model is a more accurate offline trajectory predictor, but plugging it into the current obstacle-cost guidance objective does not yet improve rollout performance.

Current interpretation:

The bottleneck is likely not only trajectory prediction accuracy. The current guidance objective / update rule may be perturbing useful OSC actions more than it helps, even when the trajectory estimator is improved.

No worker `Traceback`, `Exception`, or CUDA OOM was found in logs.
