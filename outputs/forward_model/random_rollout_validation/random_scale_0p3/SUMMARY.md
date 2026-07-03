# Forward Model Random Rollout Validation

Environment: `PickPlaceCan`
Rollouts: 3 x 100 random actions
Action scale: 0.3
Forward-model horizon: 16 steps

This validates `state + action chunk -> executed EEF xyz trajectory` directly in env rollouts.
Guidance is not involved.

Scope note: `action_scale=0.3` is a milder random-action test than the full
action-bound stress test. It is still not a policy-rollout distribution check.

## Aggregate Metrics

| predictor | traj RMSE cm | terminal mean cm | terminal p90 cm |
|---|---:|---:|---:|
| model | 2.723 +/- 0.212 | 3.310 +/- 0.340 | 5.718 +/- 0.916 |
| cumsum_action_scale_0p05 | 4.008 +/- 0.254 | 4.889 +/- 0.387 | 7.874 +/- 1.387 |

## Per-Rollout Metrics

| rollout | model traj RMSE cm | cumsum traj RMSE cm | model terminal mean cm | cumsum terminal mean cm |
|---:|---:|---:|---:|---:|
| 0 | 2.542 | 3.755 | 3.238 | 4.378 |
| 1 | 3.020 | 4.354 | 3.757 | 5.316 |
| 2 | 2.606 | 3.914 | 2.934 | 4.973 |
