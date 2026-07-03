# Forward Model Random Rollout Validation

Environment: `PickPlaceCan`
Rollouts: 3 x 100 random actions
Action scale: 1.0
Forward-model horizon: 16 steps

This validates `state + action chunk -> executed EEF xyz trajectory` directly in env rollouts.
Guidance is not involved.

Scope note: `action_scale=1.0` samples full-range random actions from the env
action bounds. This is intentionally a stress test and is likely outside the
demonstration / diffusion-policy action distribution.

## Aggregate Metrics

| predictor | traj RMSE cm | terminal mean cm | terminal p90 cm |
|---|---:|---:|---:|
| model | 6.559 +/- 0.755 | 8.382 +/- 1.501 | 13.725 +/- 1.792 |
| cumsum_action_scale_0p05 | 13.726 +/- 0.738 | 17.027 +/- 1.105 | 28.017 +/- 1.252 |

## Per-Rollout Metrics

| rollout | model traj RMSE cm | cumsum traj RMSE cm | model terminal mean cm | cumsum terminal mean cm |
|---:|---:|---:|---:|---:|
| 0 | 7.491 | 14.387 | 10.251 | 17.194 |
| 1 | 6.542 | 12.696 | 8.318 | 15.599 |
| 2 | 5.643 | 14.093 | 6.575 | 18.289 |
