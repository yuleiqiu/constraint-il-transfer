# Forward-Model Position-Only Guidance Quick Check

Date: 2026-07-02

Question:

Does limiting obstacle-guidance gradients to xyz action dimensions avoid harmful updates to rotation / gripper actions?

## Code Change

Added `--guidance_position_only`.

When enabled, the guidance gradient mask is:

```text
[1, 1, 1, 0, 0, 0, 0]
```

This means obstacle guidance can update only `action[..., :3]` and leaves rotation plus gripper dimensions unchanged.

## Setup

- Policy: `outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth`
- Environment: `PickPlaceBreadCerealCan`
- Seed: `700`
- Rollouts: `10`
- Workers: `10`
- Obstacle source: `oracle_center`
- Guidance mode: `xy`
- Trajectory backend: `forward_model`
- Forward model: `outputs/forward_model/osc_eef_forward_image_v15/model.pth`
- Guidance scale: `0.005`
- Guidance horizon: `8`

## Results

| group | guidance_position_only | success | collision_any | collision_steps | collision_rate | positive_count | positive_rate | guidance_cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_action | false | 9/10 | 0.10 | 4.5 | 0.0137 | 3.2 | 0.0766 | 0.00019089 |
| position_only | true | 9/10 | 0.20 | 10.8 | 0.0280 | 4.0 | 0.0870 | 0.00016737 |

## Conclusion

This quick check does not support the hypothesis that harmful guidance is mainly caused by modifying rotation or gripper dimensions.

The position-only mask preserved success but worsened collision metrics in this 10-rollout seed:

- collision-any increased from `0.10` to `0.20`
- collision steps increased from `4.5` to `10.8`

This is only a small quick check, but it is not promising enough to justify a full 3-seed x 50-rollout run without further diagnostics.
