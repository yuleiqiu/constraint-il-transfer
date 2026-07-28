# Guided Denoising Paired Pilot

This is a small matched-rollout study before a full evaluation matrix. Each pair runs baseline and guided policies from the same initial simulator state and policy random seed; only guidance differs.

- **Baseline:** guidance scale `0` (no guidance).
- **Guided:** `penetration` cost with guidance scale `0.01`; guidance is applied at all configured diffusion timesteps without a per-waypoint reverse-step displacement cap.
- **Clearance margin:** `0.05` m (`5` cm). The deployment cost expands each oracle obstacle's circular XY radius by this amount before testing waypoint penetration. It is a cost boundary, not a claim that the executed robot maintains that clearance.

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
| PickPlaceBreadCan | baseline | 0.700 | 0.700 | 0.050 | 0.250 | 14 | 0 | 1 | 5 |
| PickPlaceBreadCan | guided | 0.650 | 0.650 | 0.000 | 0.350 | 13 | 0 | 0 | 7 |

## Paired changes

All deltas are `guided - baseline`. The transition columns count matched initial states whose task outcome changed.

| environment | pairs | Task SR delta | CR delta | NCR delta | collision failure -> NCR | baseline success -> failure | baseline failure -> success |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCan | 20 | -0.050 | -0.050 | +0.100 | 1 | 2 | 1 |

## Guidance diagnostics

`trigger rate` is the fraction of evaluated reverse steps whose predicted trajectory enters the deployment-cost boundary. The baseline can therefore have a nonzero trigger rate even though scale `0` applies no update. Update norms are measured in normalized action space; reconstruction errors compare predicted and executed EEF trajectories. `action clips` counts policy actions outside environment limits.

| environment | condition | trigger rate | update norm mean | update norm max | max waypoint displacement (cm) | reconstruction mean (cm) | reconstruction max (cm) | observation failure rate | action clips | max clip excess |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCan | baseline | 0.109 | 0.000 | 0.000 | 0.000 | 0.253 | 11.141 | 0.000 | 0 | 1.07e-06 |
| PickPlaceBreadCan | guided | 0.079 | 0.164 | 6.009 | 3.427 | 0.282 | 16.606 | 0.000 | 0 | 1.07e-06 |

Interpret CR reduction together with Task SR and NCR. A matching NCR increase is not evidence of improved task completion.

## Decision

`reject_breadcan_5cm_xy_expansion`: The 20-pair one-distractor pilot failed the task-level gate. Task SR and Safe SR fell from 0.70 to 0.65. Guidance removed the only baseline collision (CR 0.05 to 0.00) but increased NCR from 0.25 to 0.35: one baseline collision failure became NCR, one baseline failure became success, and two baseline successes became failures. Do not scale or tune this same 5 cm XY point-guidance setting.
