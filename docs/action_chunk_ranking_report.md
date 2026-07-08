# Action-Chunk Ranking Report

Date: 2026-07-06

**Status (2026-07-08)**: archived result document. The ranking implementation
scripts have been removed; the result is retained as evidence that
geometry-only ranking on original OSC action chunks is not the active path.
`delta_eef_pose_action` replaces this route by making action chunk to EEF pose
trajectory reconstruction direct.

## Question

Does the single-object OSC diffusion policy sample geometry-safe action chunks
in multi-object obstructed scenes, and can a learned OSC forward model select
those chunks better than the old `cumsum(action * 0.05)` trajectory proxy?

## Method

Ranking replaces gradient guidance:

```text
sample K diffusion action chunks
-> predict EEF trajectory for each chunk
-> score obstacle cost / clearance
-> execute selected chunk without gradient-updating actions
```

Implemented variants:

- `cumsum` ranking: old trajectory proxy.
- `forward_model` ranking: learned `f_hat(state, OSC action chunk) -> EEF xyz`.
- `forward_model` gated ranking: keep the first sampled chunk if it is already
  safe; only rank/replace when the first chunk has positive obstacle cost.

The implementation scripts used for this report were:

```text
scripts/2026-07-03_action_chunk_ranking/run_ranking_diagnostic.py
scripts/2026-07-03_action_chunk_ranking/run_ranking_eval_matrix.py
scripts/2026-07-03_action_chunk_ranking/aggregate_ranking_results.py
```

These scripts are no longer present in the active repository.

## Same-State Diagnostic

Environment:

```text
PickPlaceBreadCerealCan
10 states
execute_horizon = 4
K = 1, 4, 8, 16
oracle obstacle centers
xy obstacle cost
```

Result at `K=16`:

| backend | actual clearance improved | mean actual clearance delta |
|---|---:|---:|
| forward model | 8/10 | +1.54 mm |
| cumsum | 4/10 | -0.74 mm |

Interpretation:

- The base policy does sample safer chunks.
- The learned forward model is a better selector than `cumsum` in the
  same-state counterfactual test.

## Rollout Pilot: Medium Environment

Environment:

```text
PickPlaceBreadCerealCan
seed = 700
10 rollouts / condition
oracle obstacle centers
xy obstacle cost
```

| condition | success | collision-any | collision steps |
|---|---:|---:|---:|
| no guidance | 0.90 | 0.50 | 18.80 |
| cumsum ranking K=16 | 0.80 | 0.40 | 5.10 |
| forward-model ranking K=16 | 0.60 | 0.10 | 1.10 |
| forward-model ranking K=4 | 0.80 | 0.30 | 2.20 |
| forward-model ranking K=16 gated | 0.90 | 0.30 | 38.00 |

Interpretation:

- Ranking can reduce collision-any.
- Pure geometry ranking can over-optimize clearance and hurt task success.
- Gating restores success in this small pilot but does not reliably reduce
  collision duration.

## Rollout Pilot: Hardest Environment

Environment:

```text
PickPlaceBreadCerealMilkCan
seeds = 700, 701, 702
20 rollouts / seed
60 rollouts / condition
oracle obstacle centers
xy obstacle cost
```

| condition | success | collision-any | collision steps |
|---|---:|---:|---:|
| no guidance | 53/60 = 0.883 | 0.417 | 23.63 |
| forward-model ranking K=4 | 35/60 = 0.583 | 0.383 | 47.27 |
| forward-model ranking K=16 gated | 50/60 = 0.833 | 0.367 | 36.90 |

Interpretation:

- The oracle-mask baseline is already high even in the hardest environment.
- Forward-model ranking slightly reduces collision-any but hurts success and
  increases collision duration.
- Current geometry-only ranking is not a successful rollout method.

Historical baseline context:

```text
PickPlaceBreadCerealMilkCan no-guidance baseline:
127/150 = 0.847 success in the prior 600-rollout evaluation
```

So the high no-guidance success rate is not just a seed-700 artifact.

## Conclusion

This experiment answers the immediate handoff question:

```text
The single-object policy does sample geometry-safer chunks, and the forward
model can recover them better than cumsum in same-state diagnostics.
```

But it also shows:

```text
Geometry-only action-chunk ranking is not enough to improve rollout success.
```

The failure mode is not lack of safe samples. It is scoring: selecting for
obstacle clearance alone can choose chunks that are less task-preserving, harm
grasp / placement timing, and increase collision duration in some failures.

## Next Step

Do not continue with geometry-only ranking as the final method. If this route
continues, the ranking objective needs a task-preserving term, for example:

- obstacle cost as the primary hard filter;
- distance to the first sampled chunk as a tie-break / regularizer;
- policy likelihood or denoising score as a prior;
- task-progress heuristic near grasp / place phases.

Before larger rollout matrices, re-check whether failures are actually caused
by non-target collisions, since the oracle-mask baseline is already high and
collision-any is not tightly coupled to success.
