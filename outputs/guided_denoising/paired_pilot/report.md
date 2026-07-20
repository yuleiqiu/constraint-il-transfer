# Guided Denoising Paired Pilot

This is a small matched-rollout study before a full evaluation matrix. Each pair runs baseline and guided policies from the same initial simulator state and policy random seed; only guidance differs.

- **Baseline:** guidance scale `0` (no guidance).
- **Guided:** guidance scale `0.01`. This is the base physical waypoint-push coefficient $\lambda$; the actual per-step displacement also includes the DDIM timestep factor $1 / \sqrt{\bar{\alpha}_{t-1}}$, so this value is not a fixed displacement at every denoising step.
- **Clearance margin:** `0.02` m (`2` cm). The deployment cost expands each oracle obstacle's circular XY radius by this amount before testing waypoint penetration. It is a cost boundary, not a claim that the executed robot maintains that clearance.

## Metric definitions

- **Task SR (Task Success Rate):** fraction of episodes that complete the environment task, whether or not a distractor collision occurred.
- **Safe SR (Safe Success Rate):** fraction of episodes that complete the task with no robot contact against any non-target distractor.
- **CR (Collision Rate):** fraction of episodes with at least one robot-distractor contact, including both successful and failed episodes.
- **NCR (collision-free Non-Completion Rate):** fraction of episodes that do not collide but still fail to complete the task.

The final four columns below are mutually exclusive episode counts and sum to the number of episodes: safe success, success with collision, collision failure, and collision-free non-completion.

For rates, `Task SR = safe success rate + success-with-collision rate`; a reduction in CR is useful only if it does not merely move collision failures into NCR.

## Outcome metrics

| environment | condition | Task SR | Safe SR | CR | NCR | safe success | success + collision | collision failure | NCR count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | baseline | 0.100 | 0.100 | 0.800 | 0.100 | 1 | 0 | 8 | 1 |
| PickPlaceBreadCerealCan | guided | 0.200 | 0.200 | 0.600 | 0.200 | 2 | 0 | 6 | 2 |
| PickPlaceBreadCerealMilkCan | baseline | 0.200 | 0.200 | 0.700 | 0.100 | 2 | 0 | 7 | 1 |
| PickPlaceBreadCerealMilkCan | guided | 0.300 | 0.100 | 0.900 | 0.000 | 1 | 2 | 7 | 0 |

## Paired changes

All deltas are `guided - baseline`. The transition columns count matched initial states whose task outcome changed.

| environment | pairs | Task SR delta | CR delta | NCR delta | collision failure -> NCR | baseline success -> failure | baseline failure -> success |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | 10 | +0.100 | -0.200 | +0.100 | 1 | 0 | 1 |
| PickPlaceBreadCerealMilkCan | 10 | +0.100 | +0.200 | -0.100 | 0 | 1 | 2 |

## Guidance diagnostics

`trigger rate` is the fraction of DDIM reverse steps whose predicted trajectory enters the deployment-cost boundary. The baseline can therefore have a nonzero trigger rate even though scale `0` applies no update. Update norms are measured in normalized action space; reconstruction errors compare predicted and executed EEF trajectories. `action clips` counts policy actions outside environment limits.

| environment | condition | trigger rate | update norm mean | update norm max | reconstruction mean (cm) | reconstruction max (cm) | action clips |
|---|---|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | baseline | 0.480 | 0.000 | 0.000 | 0.328 | 11.455 | 0 |
| PickPlaceBreadCerealCan | guided | 0.283 | 0.625 | 10.172 | 0.326 | 9.701 | 0 |
| PickPlaceBreadCerealMilkCan | baseline | 0.270 | 0.000 | 0.000 | 0.341 | 15.973 | 0 |
| PickPlaceBreadCerealMilkCan | guided | 0.241 | 0.545 | 5.982 | 0.328 | 12.141 | 0 |

Interpret CR reduction together with Task SR and NCR. A matching NCR increase is not evidence of improved task completion.

## Decision

`do_not_scale_to_full_matrix`: The 20-pair pilot raises Task SR by 0.10 in both environments, but the safety effect is inconsistent: CR falls by 0.20 while NCR rises by 0.10 in PickPlaceBreadCerealCan, whereas CR rises by 0.20 and Safe SR falls by 0.10 in PickPlaceBreadCerealMilkCan. This does not satisfy the Gate D criterion for scaling the current setting.
