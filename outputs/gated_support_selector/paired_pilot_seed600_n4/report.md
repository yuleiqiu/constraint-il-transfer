# Gated Support Selector Paired Rollout

The four-way partition is safe success / success with collision / collision failure / collision-free non-completion.

| env | condition | n | Task SR | Safe SR | CR | NCR | triggered / corrective / fallback chunks |
|---|---|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | source | 4 | 0.250 | 0.250 | 0.750 | 0.000 | 0 / 0 / 0 |
| PickPlaceBreadCerealCan | gated | 4 | 0.250 | 0.250 | 0.750 | 0.000 | 2 / 1 / 1 |
| PickPlaceCan | source | 4 | 0.750 | 0.750 | 0.000 | 0.250 | 0 / 0 / 0 |
| PickPlaceCan | gated | 4 | 0.750 | 0.750 | 0.000 | 0.250 | 0 / 0 / 0 |

## Paired transitions

| env | source | gated | count |
|---|---|---|---:|
| PickPlaceBreadCerealCan | collision_failure | collision_failure | 3 |
| PickPlaceBreadCerealCan | safe_success | safe_success | 1 |
| PickPlaceCan | ncr | ncr | 1 |
| PickPlaceCan | safe_success | safe_success | 3 |
