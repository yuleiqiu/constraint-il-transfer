# Full50 local planner comparison

Complete previously examined50-layout bank. One rollout per condition/layout; no held-out claim.
Criterion: old strict success OR released placement with0.5s residence.

| Layout | Baseline task / safe | Planner task / safe | Planner changed chunks | Handoffs |
|---|---|---|---:|---:|
| 0 | 1 / 1 | 1 / 1 | 11 | 1 |
| 1 | 1 / 1 | 1 / 1 | 12 | 1 |
| 2 | 1 / 1 | 1 / 1 | 10 | 1 |
| 3 | 1 / 1 | 1 / 1 | 0 | 0 |
| 4 | 1 / 1 | 1 / 1 | 11 | 1 |
| 5 | 1 / 1 | 1 / 1 | 9 | 1 |
| 6 | 1 / 1 | 1 / 1 | 10 | 1 |
| 7 | 1 / 1 | 1 / 1 | 11 | 1 |
| 8 | 1 / 0 | 1 / 1 | 13 | 1 |
| 9 | 1 / 0 | 1 / 1 | 13 | 1 |
| 10 | 1 / 1 | 1 / 1 | 10 | 1 |
| 11 | 1 / 1 | 1 / 1 | 9 | 1 |
| 12 | 1 / 1 | 1 / 1 | 10 | 1 |
| 13 | 1 / 1 | 1 / 1 | 0 | 0 |
| 14 | 0 / 0 | 1 / 1 | 11 | 1 |
| 15 | 1 / 1 | 1 / 1 | 11 | 1 |
| 16 | 1 / 1 | 1 / 1 | 7 | 1 |
| 17 | 1 / 1 | 1 / 1 | 13 | 1 |
| 18 | 1 / 0 | 1 / 1 | 12 | 1 |
| 19 | 1 / 1 | 1 / 1 | 11 | 1 |
| 20 | 1 / 1 | 1 / 1 | 0 | 0 |
| 21 | 1 / 1 | 1 / 1 | 7 | 1 |
| 22 | 1 / 1 | 1 / 1 | 12 | 1 |
| 23 | 1 / 1 | 1 / 1 | 12 | 1 |
| 24 | 1 / 1 | 1 / 1 | 0 | 0 |
| 25 | 1 / 0 | 1 / 1 | 10 | 1 |
| 26 | 1 / 1 | 1 / 1 | 10 | 1 |
| 27 | 0 / 0 | 0 / 0 | 13 | 1 |
| 28 | 1 / 0 | 1 / 1 | 12 | 1 |
| 29 | 1 / 1 | 1 / 1 | 10 | 1 |
| 30 | 1 / 1 | 1 / 1 | 0 | 0 |
| 31 | 1 / 1 | 1 / 1 | 12 | 1 |
| 32 | 1 / 1 | 1 / 1 | 10 | 1 |
| 33 | 1 / 1 | 1 / 1 | 10 | 1 |
| 34 | 0 / 0 | 1 / 1 | 10 | 1 |
| 35 | 1 / 1 | 1 / 1 | 9 | 1 |
| 36 | 1 / 1 | 1 / 1 | 12 | 1 |
| 37 | 1 / 0 | 1 / 1 | 13 | 1 |
| 38 | 1 / 1 | 1 / 1 | 9 | 1 |
| 39 | 1 / 1 | 1 / 1 | 9 | 1 |
| 40 | 1 / 1 | 1 / 1 | 11 | 1 |
| 41 | 1 / 1 | 1 / 1 | 12 | 1 |
| 42 | 1 / 1 | 1 / 1 | 12 | 1 |
| 43 | 0 / 0 | 1 / 1 | 12 | 1 |
| 44 | 1 / 1 | 1 / 1 | 11 | 1 |
| 45 | 1 / 1 | 1 / 1 | 12 | 1 |
| 46 | 1 / 1 | 1 / 1 | 10 | 1 |
| 47 | 1 / 1 | 1 / 1 | 0 | 0 |
| 48 | 1 / 1 | 1 / 1 | 11 | 1 |
| 49 | 1 / 0 | 1 / 1 | 12 | 1 |

Task-level engineering gate: **PASS**.

See summary.json for Task/Safe/CR/NCR, four-way partitions and paired transitions.
See video_index.md for all actual-execution MP4s. No independent held-out validation is implied.
