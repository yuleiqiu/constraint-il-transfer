# Delta EEF Policy Multi-Environment Evaluation

Date: 2026-07-14

## Scope

This evaluation characterizes one trained clean-image delta EEF diffusion
policy checkpoint across the four PickPlace environments. It is not a
three-training-seed policy benchmark.

Checkpoint:

```text
outputs/robomimic/train/dp_can_delta_pose_osc/20260707222943/models/model_epoch_260_image_v15_delta_eef_pose_osc_success_0.98.pth
```

Evaluation protocol:

- environments: PickPlaceCan plus 1, 2, and 3 distractor variants;
- evaluation seeds: 600, 601, 602;
- episodes: 50 per environment and seed, 600 total;
- horizon: 400;
- action interface: delta EEF full pose through `OSC_POSE` in world frame.

## Core Results

Task SR is the standard task success rate. Safe SR counts only successful
episodes without non-target contact. CR counts any episode with non-target
contact. NCR counts collision-free episodes that do not complete within the
horizon. Safe SR, CR, and NCR are mutually exclusive and sum to one.

| environment | Task SR | Safe SR | CR | NCR |
|---|---:|---:|---:|---:|
| PickPlaceCan | 0.907 +/- 0.023 | 0.907 +/- 0.023 | 0.000 +/- 0.000 | 0.093 +/- 0.023 |
| PickPlaceBreadCan | 0.707 +/- 0.114 | 0.687 +/- 0.129 | 0.213 +/- 0.058 | 0.100 +/- 0.072 |
| PickPlaceBreadCerealCan | 0.253 +/- 0.070 | 0.240 +/- 0.060 | 0.647 +/- 0.050 | 0.113 +/- 0.012 |
| PickPlaceBreadCerealMilkCan | 0.200 +/- 0.035 | 0.153 +/- 0.023 | 0.680 +/- 0.060 | 0.167 +/- 0.064 |

Values are mean +/- sample standard deviation over the three evaluation seeds.

For diagnosis, retain the complete four-way partition rather than merging all
collision episodes into CR:

| environment | safe success | success with collision | collision failure | NCR |
|---|---:|---:|---:|---:|
| PickPlaceCan | 0.907 | 0.000 | 0.000 | 0.093 |
| PickPlaceBreadCan | 0.687 | 0.020 | 0.193 | 0.100 |
| PickPlaceBreadCerealCan | 0.240 | 0.013 | 0.633 | 0.113 |
| PickPlaceBreadCerealMilkCan | 0.153 | 0.047 | 0.633 | 0.167 |

## Failure Analysis

Physical obstruction is the dominant failure mode in the harder environments,
but collision is not equivalent to failure.

| environment | failures with collision | collision before target | target never contacted |
|---|---:|---:|---:|
| PickPlaceBreadCan | 65.9% | 52.3% | 72.7% |
| PickPlaceBreadCerealCan | 84.8% | 81.2% | 83.0% |
| PickPlaceBreadCerealMilkCan | 79.2% | 79.2% | 80.0% |

The representative trajectories show three distinct behaviors:

- pre-target obstruction: the EEF contacts a distractor before reaching Can,
  and Can remains nearly stationary;
- collision-free task failure: some episodes grasp and move Can but fail the
  final placement, so obstacle avoidance alone cannot recover them;
- incidental collision: some episodes contact or move a distractor and still
  complete the task, so collision-any is not a sufficient failure label.

Representative cases:

- [Pre-target obstruction](multienv_eval_cases/geo_blocked_before_target.png)
- [Collision-free placement failure](multienv_eval_cases/no_collision_place_failure.png)
- [Success with collision](multienv_eval_cases/success_with_collision.png)
- [Clean success](multienv_eval_cases/clean_success.png)

## Action Interface Health

The evaluation did not reveal an action-interface regression:

| environment | action clips | mean position reconstruction error | mean orientation reconstruction error |
|---|---:|---:|---:|
| PickPlaceCan | 0 | 0.150 cm | 0.212 deg |
| PickPlaceBreadCan | 0 | 0.178 cm | 0.272 deg |
| PickPlaceBreadCerealCan | 0 | 0.354 cm | 0.705 deg |
| PickPlaceBreadCerealMilkCan | 0 | 0.375 cm | 0.719 deg |

The policy-predicted delta EEF chunks therefore remain accurate enough to use
directly for trajectory scoring. The current multi-object failures are not a
return of the old OSC action-to-trajectory mapping problem.

## Implications

- Future obstacle-aware evaluation should report Task SR, Safe SR, CR, and NCR.
- CR reduction is only useful if it does not produce a matching NCR increase.
- Failure analysis should preserve safe success, success with collision,
  collision failure, and NCR as a four-way partition.
- Geometry-aware ranking should focus on pre-grasp / pre-target obstruction;
  it should not treat every distractor contact as equally fatal.
- Placement failures require a separate task-phase remedy and cannot be solved
  by obstacle avoidance alone.

The raw HDF5 trajectories remain local under the ignored eval directory. The
analysis and plotting scripts reproduce the derived tables and case plots.
