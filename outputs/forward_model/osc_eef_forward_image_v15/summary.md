# OSC EEF Forward Model

Dataset: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15.hdf5`
DP config: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/robomimic/exps/baseline/diffusion_policy_can_masked_image.json`
Resolved horizon: `16`

## Validation Metrics

| predictor | traj RMSE cm | terminal mean cm | terminal median cm | terminal p90 cm |
|---|---:|---:|---:|---:|
| model | 0.264 | 0.280 | 0.222 | 0.522 |
| hold | 3.930 | 5.873 | 6.357 | 9.088 |
| cumsum_action_scale_0p05 | 11.831 | 17.560 | 18.737 | 27.363 |
| cumsum_fitted_scalar | 0.956 | 0.956 | 0.632 | 1.705 |
| cumsum_fitted_diagonal | 0.917 | 0.936 | 0.646 | 1.592 |
| cumsum_fitted_full_linear | 0.857 | 0.917 | 0.649 | 1.586 |

## Interpretation

- Model trajectory RMSE improvement over old `action * 0.05` cumsum: `44.74x`.
- Model terminal error improvement over old `action * 0.05` cumsum: `62.82x`.
- Best fitted cumsum baseline by trajectory RMSE: `cumsum_fitted_full_linear` at `0.857 cm`; model improves this by `3.25x`.
- Best fitted cumsum baseline by terminal error: `cumsum_fitted_full_linear` at `0.917 cm`; model improves this by `3.28x`.
- This model is only a guidance surrogate. It does not replace the OSC controller or the diffusion policy.

Scope note:

- These numbers are from held-out demonstration windows, not arbitrary control
  inputs.
- Random-action env rollout validation was added later. It still shows the
  model beats `cumsum(action * 0.05)`, but with larger absolute errors under
  out-of-distribution random actions.
- Rollout-level gradient guidance with this model did not improve success rate
  in the current implementation, so model accuracy should not be conflated with
  guidance usefulness.

## Config

```json
{
  "dataset": "third_party/robomimic/datasets/can/yq/image_v15.hdf5",
  "dp_config": "third_party/robomimic/robomimic/exps/baseline/diffusion_policy_can_masked_image.json",
  "output_dir": "outputs/forward_model/osc_eef_forward_image_v15",
  "horizon": "auto",
  "state_keys": [
    "obs/robot0_eef_pos",
    "obs/robot0_eef_quat",
    "obs/robot0_gripper_qpos"
  ],
  "action_key": "actions",
  "target_key": "next_obs/robot0_eef_pos",
  "model": {
    "hidden_dim": 512,
    "state_embed_dim": 128,
    "action_embed_dim": 256,
    "dropout": 0.0
  },
  "train": {
    "batch_size": 1024,
    "epochs": 200,
    "lr": 0.001,
    "weight_decay": 1e-05,
    "terminal_weight": 2.0,
    "patience": 30,
    "seed": 0,
    "num_workers": 0
  },
  "resolved_horizon": 16,
  "resolved_dataset": "/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15.hdf5",
  "resolved_dp_config": "/home/yulei/codes/constraint-il-transfer/third_party/robomimic/robomimic/exps/baseline/diffusion_policy_can_masked_image.json"
}
```
