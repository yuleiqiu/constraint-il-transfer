# Gated Support Selector Paired Rollout

The four-way partition is safe success / success with collision / collision failure / collision-free non-completion.

| env | condition | n | Task SR | Safe SR | CR | NCR | triggered / corrective / fallback chunks |
|---|---|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | source | 20 | 0.250 | 0.150 | 0.800 | 0.050 | 0 / 0 / 0 |
| PickPlaceBreadCerealCan | gated | 20 | 0.150 | 0.100 | 0.750 | 0.150 | 13 / 5 / 8 |

## Paired transitions

| env | source | gated | count |
|---|---|---|---:|
| PickPlaceBreadCerealCan | collision_failure | collision_failure | 14 |
| PickPlaceBreadCerealCan | ncr | ncr | 1 |
| PickPlaceBreadCerealCan | safe_success | ncr | 1 |
| PickPlaceBreadCerealCan | safe_success | safe_success | 2 |
| PickPlaceBreadCerealCan | success_with_collision | ncr | 1 |
| PickPlaceBreadCerealCan | success_with_collision | success_with_collision | 1 |
