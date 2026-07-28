# Guided Denoising Paired Pilot

This is a small matched-rollout study before a full evaluation matrix. Each pair runs baseline and guided policies from the same initial simulator state and policy random seed; only guidance differs.

- **Baseline:** guidance scale `0` (no guidance).
- **Guided:** `penetration` cost with guidance scale `0.003`; guidance is applied at diffusion timesteps `t <= 19` with a per-waypoint reverse-step displacement cap of `0.03` m.
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
| PickPlaceBreadCan | baseline | 0.450 | 0.400 | 0.050 | 0.550 | 8 | 1 | 0 | 11 |
| PickPlaceBreadCan | guided | 0.450 | 0.350 | 0.150 | 0.500 | 7 | 2 | 1 | 10 |

## Paired changes

All deltas are `guided - baseline`. The transition columns count matched initial states whose task outcome changed.

| environment | pairs | Task SR delta | CR delta | NCR delta | collision failure -> NCR | baseline success -> failure | baseline failure -> success |
|---|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCan | 20 | +0.000 | +0.100 | -0.050 | 0 | 1 | 1 |

## Guidance diagnostics

`trigger rate` is the fraction of evaluated reverse steps whose predicted trajectory enters the deployment-cost boundary. The baseline can therefore have a nonzero trigger rate even though scale `0` applies no update. Update norms are measured in normalized action space; reconstruction errors compare predicted and executed EEF trajectories. `action clips` counts policy actions outside environment limits.

| environment | condition | trigger rate | update norm mean | update norm max | max waypoint displacement (cm) | reconstruction mean (cm) | reconstruction max (cm) | observation failure rate | action clips | max clip excess |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCan | baseline | 0.045 | 0.000 | 0.000 | 0.000 | 0.202 | 9.322 | 0.000 | 3225 | 1.50e-05 |
| PickPlaceBreadCan | guided | 0.034 | 0.017 | 1.103 | 0.315 | 0.225 | 9.322 | 0.000 | 3235 | 1.50e-05 |

Interpret CR reduction together with Task SR and NCR. A matching NCR increase is not evidence of improved task completion.

## Decision

`reject_lan_o3dp_epoch350_breadcan_guidance`: The frozen point-cloud LAN-O3DP configuration did not improve PickPlaceBreadCan over 20 matched pairs. Task SR stayed at 0.45, Safe SR fell from 0.40 to 0.35, and CR rose from 0.05 to 0.15. One baseline failure became a guided success with collision, while one baseline safe success became a guided collision failure. Do not scale this configuration.

## Synchronized videos

Two outcome-changing pairs were replayed from their exact recorded simulator
states. State hashes and replay outcomes were verified before accepting the
videos. Both pairs also have four-view variants showing the two-frame
oracle-segmented policy point cloud below the synchronized RGB views. See
`videos/README.md`.
