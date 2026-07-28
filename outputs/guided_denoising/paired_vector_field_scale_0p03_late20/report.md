# Guided Denoising Paired Pilot

This is a small matched-rollout study before a full evaluation matrix. Each pair runs baseline and guided policies from the same initial simulator state and policy random seed; only guidance differs.

- **Baseline:** guidance scale `0` (no guidance).
- **Guided:** `vector_field` cost with guidance scale `0.03`; guidance is applied at diffusion timesteps `t <= 19` with a per-waypoint reverse-step displacement cap of `0.03` m.
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
| PickPlaceBreadCerealCan | baseline | 0.600 | 0.400 | 0.500 | 0.100 | 4 | 2 | 3 | 1 |
| PickPlaceBreadCerealCan | guided | 0.500 | 0.400 | 0.600 | 0.000 | 4 | 1 | 5 | 0 |
| PickPlaceBreadCerealMilkCan | baseline | 0.800 | 0.600 | 0.400 | 0.000 | 6 | 2 | 2 | 0 |
| PickPlaceBreadCerealMilkCan | guided | 0.700 | 0.500 | 0.400 | 0.100 | 5 | 2 | 2 | 1 |

## Paired changes

All deltas are `guided - baseline`. The transition columns count matched initial states whose task outcome changed.

| environment | pairs | Task SR delta | CR delta | NCR delta | collision failure -> NCR | baseline success -> failure | baseline failure -> success |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | 10 | -0.100 | +0.100 | -0.100 | 0 | 1 | 0 |
| PickPlaceBreadCerealMilkCan | 10 | -0.100 | +0.000 | +0.100 | 0 | 1 | 0 |

## Guidance diagnostics

`trigger rate` is the fraction of evaluated reverse steps whose predicted trajectory enters the deployment-cost boundary. The baseline can therefore have a nonzero trigger rate even though scale `0` applies no update. Update norms are measured in normalized action space; reconstruction errors compare predicted and executed EEF trajectories. `action clips` counts policy actions outside environment limits.

| environment | condition | trigger rate | update norm mean | update norm max | max waypoint displacement (cm) | reconstruction mean (cm) | reconstruction max (cm) | observation failure rate | action clips | max clip excess |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | baseline | 0.057 | 0.000 | 0.000 | 0.000 | 0.312 | 13.642 | 0.000 | 1515 | 1.50e-05 |
| PickPlaceBreadCerealCan | guided | 0.099 | 0.031 | 1.435 | 2.458 | 0.426 | 14.958 | 0.000 | 1482 | 1.50e-05 |
| PickPlaceBreadCerealMilkCan | baseline | 0.111 | 0.000 | 0.000 | 0.000 | 0.318 | 16.717 | 0.000 | 1348 | 1.50e-05 |
| PickPlaceBreadCerealMilkCan | guided | 0.081 | 0.029 | 1.304 | 1.966 | 0.306 | 9.525 | 0.000 | 1482 | 1.50e-05 |

Interpret CR reduction together with Task SR and NCR. A matching NCR increase is not evidence of improved task completion.

## Decision

`stop_point_level_guidance`: The 20-pair pilot failed the fixed gate. Task SR fell by 0.10 in both environments; CR increased by 0.10 in PickPlaceBreadCerealCan and was unchanged in PickPlaceBreadCerealMilkCan. No baseline failure became a guided success, while two baseline successes became guided failures. Stop tuning point-level EEF guidance and move to a gripper-aware geometry formulation.
