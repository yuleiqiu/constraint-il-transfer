# Sample-Refine Paired Rollout Report

The four-way partition is safe success / success with collision / collision failure / collision-free non-completion (NCR).

| env | condition | n | Task SR | Safe SR | CR | NCR | safe / succ+coll / coll fail / NCR | ms/chunk | overhead | coverage | refine accept/fallback | clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | baseline | 20 | 0.100 | 0.050 | 0.750 | 0.200 | 1 / 1 / 14 / 4 | 30.352 | 1.000x | 0.409 | 0/0 | 0 |
| PickPlaceBreadCerealCan | refine_only | 20 | 0.100 | 0.050 | 0.600 | 0.350 | 1 / 1 / 11 / 7 | 53.765 | 1.771x | 0.456 | 160/823 | 0 |
| PickPlaceBreadCerealCan | sample_only | 20 | 0.150 | 0.100 | 0.750 | 0.150 | 2 / 1 / 14 / 3 | 30.320 | 0.999x | 0.455 | 0/0 | 0 |
| PickPlaceBreadCerealCan | sample_refine | 20 | 0.100 | 0.050 | 0.750 | 0.200 | 1 / 1 / 14 / 4 | 53.303 | 1.756x | 0.398 | 188/793 | 0 |

## Baseline P0 critical-regime stratification

Each matched layout uses the baseline episode's canonical P0 critical chunk as a shared label across all four conditions.

| env | condition | baseline critical regime | n | Task SR | Safe SR | CR | NCR |
|---|---|---|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | baseline | non_target_directed | 9 | 0.111 | 0.000 | 0.778 | 0.222 |
| PickPlaceBreadCerealCan | baseline | target_directed_clear | 4 | 0.250 | 0.250 | 0.250 | 0.500 |
| PickPlaceBreadCerealCan | baseline | target_directed_obstructed | 7 | 0.000 | 0.000 | 1.000 | 0.000 |
| PickPlaceBreadCerealCan | refine_only | non_target_directed | 9 | 0.111 | 0.000 | 0.556 | 0.444 |
| PickPlaceBreadCerealCan | refine_only | target_directed_clear | 4 | 0.250 | 0.250 | 0.250 | 0.500 |
| PickPlaceBreadCerealCan | refine_only | target_directed_obstructed | 7 | 0.000 | 0.000 | 0.857 | 0.143 |
| PickPlaceBreadCerealCan | sample_only | non_target_directed | 9 | 0.111 | 0.000 | 0.778 | 0.222 |
| PickPlaceBreadCerealCan | sample_only | target_directed_clear | 4 | 0.500 | 0.500 | 0.250 | 0.250 |
| PickPlaceBreadCerealCan | sample_only | target_directed_obstructed | 7 | 0.000 | 0.000 | 1.000 | 0.000 |
| PickPlaceBreadCerealCan | sample_refine | non_target_directed | 9 | 0.111 | 0.000 | 0.778 | 0.222 |
| PickPlaceBreadCerealCan | sample_refine | target_directed_clear | 4 | 0.250 | 0.250 | 0.250 | 0.500 |
| PickPlaceBreadCerealCan | sample_refine | target_directed_obstructed | 7 | 0.000 | 0.000 | 1.000 | 0.000 |

## Paired outcome transitions

| env | condition | baseline outcome | condition outcome | count |
|---|---|---|---|---:|
| PickPlaceBreadCerealCan | sample_only | collision_failure | collision_failure | 14 |
| PickPlaceBreadCerealCan | sample_only | ncr | ncr | 3 |
| PickPlaceBreadCerealCan | sample_only | ncr | safe_success | 1 |
| PickPlaceBreadCerealCan | sample_only | safe_success | safe_success | 1 |
| PickPlaceBreadCerealCan | sample_only | success_with_collision | success_with_collision | 1 |
| PickPlaceBreadCerealCan | refine_only | collision_failure | collision_failure | 11 |
| PickPlaceBreadCerealCan | refine_only | collision_failure | ncr | 3 |
| PickPlaceBreadCerealCan | refine_only | ncr | ncr | 4 |
| PickPlaceBreadCerealCan | refine_only | safe_success | safe_success | 1 |
| PickPlaceBreadCerealCan | refine_only | success_with_collision | success_with_collision | 1 |
| PickPlaceBreadCerealCan | sample_refine | collision_failure | collision_failure | 14 |
| PickPlaceBreadCerealCan | sample_refine | ncr | ncr | 4 |
| PickPlaceBreadCerealCan | sample_refine | safe_success | safe_success | 1 |
| PickPlaceBreadCerealCan | sample_refine | success_with_collision | success_with_collision | 1 |

## Development gate

- **PickPlaceBreadCerealCan: `do_not_advance`.** sample_refine did not improve paired Task SR. Task SR Δ=+0.000, Safe SR Δ=+0.000, CR Δ=+0.000, NCR Δ=+0.000.
