# Delta EEF Multi-Env Eval Analysis

Eval dir: `outputs/targeted_finetuning/checkpoint_screen_seed600_n20/mixed_u02000`

## Environment Summary

| env | success mean±std | collision mean±std | failures | failure+collision | collision before target | never target contact | touched no lift | lifted not success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | 0.050±0.000 | 0.300±0.000 | 19 | 6 (31.6%) | 6 (31.6%) | 15 (78.9%) | 0 | 4 |
| PickPlaceCan | 1.000±0.000 | 0.000±0.000 | 0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 0 |

## Safety / Completion Partition

Paper metrics use safe success as SR, any non-target collision as CR, and collision-free failure as NCR. By construction, safe SR + CR + NCR = 1.

| env | task SR | safe SR mean±std | CR mean±std | NCR mean±std | success+collision | collision failure |
|---|---:|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | 0.050 | 0.050±0.000 | 0.300±0.000 | 0.650±0.000 | 0.000 | 0.300 |
| PickPlaceCan | 1.000 | 1.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000 | 0.000 |

## Trajectory / Controller Health

| env | action clips | mean chunk pos err cm | mean chunk ori err deg | median first target step | median first collision step |
|---|---:|---:|---:|---:|---:|
| PickPlaceBreadCerealCan | 0 | 0.623 | 0.934 | 184.0 | 135.0 |
| PickPlaceCan | 0 | 0.163 | 0.171 | 112.0 | n/a |

## Clearance / Lift

| env | median min clearance success m | median min clearance failure m | mean target z gain success m | mean target z gain failure m | contact names |
|---|---:|---:|---:|---:|---|
| PickPlaceBreadCerealCan | 0.007 | -0.032 | 0.272 | 0.048 | `{"Cereal": 6}` |
| PickPlaceCan | n/a | n/a | 0.203 | n/a | `{}` |

## Reading

- `failure+collision` is an overlap metric: it counts failed episodes with any non-target contact.
- The four-way partition is safe success / success with collision / collision failure / NCR.
- `collision before target` is a stronger obstruction signal: the first non-target contact happened before the first target contact, or the target was never contacted.
- `touched no lift` uses target max z gain < 0.05 m.
- Negative clearance means the coarse EEF-obstacle clearance proxy crossed the obstacle radius.
