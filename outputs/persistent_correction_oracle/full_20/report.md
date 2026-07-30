# Persistent-Correction Oracle: Full 20-State Evaluation

The existing SDP-6k policy is executed for multiple consecutive closed-loop chunks. Every collision-free chunk boundary is cloned and handed to the frozen source policy.

## Aggregate

- states / trajectories: 20 / 80
- safe-prefix temporal boundaries: 1099
- safe recoverable handoffs: 311
- states with a safe recoverable handoff: 7
- trajectories with a safe recoverable handoff: 27
- direct safe corrective successes: 0
- source-continuation outcomes: `{'collision_failure': 280, 'ncr': 488, 'safe_success': 311, 'success_with_collision': 20}`
- existence-gate decision: **PASS**

## Per state

| state | trajectories | boundaries | trajectories with safe handoff | earliest trajectory / chunk / actions / absolute step |
|---|---:|---:|---:|---|
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_000 | 4 | 80 | 0 | -- |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_001 | 4 | 66 | 3 | 1 / 5 / 40 / 64 |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_002 | 4 | 73 | 0 | -- |
| one_chunk_lead_states_seed601_602:PickPlaceBreadCerealCan_preentry_000 | 4 | 80 | 0 | -- |
| one_chunk_lead_states_seed601_602:PickPlaceBreadCerealCan_preentry_001 | 4 | 23 | 0 | -- |
| one_chunk_lead_states_seed601_602:PickPlaceBreadCerealCan_preentry_002 | 4 | 80 | 4 | 0 / 1 / 8 / 88 |
| one_chunk_lead_states_seed601_602:PickPlaceBreadCerealCan_preentry_003 | 4 | 80 | 4 | 0 / 6 / 48 / 120 |
| one_chunk_lead_states_seed601_602:PickPlaceBreadCerealCan_preentry_004 | 4 | 4 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_000 | 4 | 80 | 4 | 2 / 2 / 16 / 72 |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_001 | 4 | 80 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_002 | 4 | 8 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_003 | 4 | 23 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_004 | 4 | 80 | 4 | 1 / 6 / 48 / 120 |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_005 | 4 | 18 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_006 | 4 | 58 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_007 | 4 | 80 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_008 | 4 | 21 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_009 | 4 | 5 | 0 | -- |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_010 | 4 | 80 | 4 | 0 / 1 / 8 / 64 |
| one_chunk_lead_states_seed603_607:PickPlaceBreadCerealCan_preentry_011 | 4 | 80 | 4 | 0 / 6 / 48 / 120 |

Trajectory indices below are zero-based. Counts over temporal boundaries are descriptive, not independent trials.

## Recoverability windows

| state | trajectory | safe chunk windows | correction ending |
|---|---:|---|---|
| 0 | 0 | -- | maximum duration |
| 0 | 1 | -- | maximum duration |
| 0 | 2 | -- | maximum duration |
| 0 | 3 | -- | maximum duration |
| 1 | 0 | 7-9, 11, 13-20 | maximum duration |
| 1 | 1 | 5-7, 14-20 | maximum duration |
| 1 | 2 | -- | collision after 6 safe boundaries |
| 1 | 3 | 6-7, 13-20 | maximum duration |
| 2 | 0 | -- | maximum duration |
| 2 | 1 | -- | collision after 13 safe boundaries |
| 2 | 2 | -- | maximum duration |
| 2 | 3 | -- | maximum duration |
| 3 | 0 | -- | maximum duration |
| 3 | 1 | -- | maximum duration |
| 3 | 2 | -- | maximum duration |
| 3 | 3 | -- | maximum duration |
| 4 | 0 | -- | collision after 1 safe boundaries |
| 4 | 1 | -- | collision after 1 safe boundaries |
| 4 | 2 | -- | collision after 1 safe boundaries |
| 4 | 3 | -- | maximum duration |
| 5 | 0 | 1-20 | maximum duration |
| 5 | 1 | 1-2, 4-20 | maximum duration |
| 5 | 2 | 1-20 | maximum duration |
| 5 | 3 | 1-7 | maximum duration |
| 6 | 0 | 6-20 | maximum duration |
| 6 | 1 | 7-10, 13 | maximum duration |
| 6 | 2 | 6-8 | maximum duration |
| 6 | 3 | 7-8 | maximum duration |
| 7 | 0 | -- | collision after 1 safe boundaries |
| 7 | 1 | -- | collision after 1 safe boundaries |
| 7 | 2 | -- | collision after 1 safe boundaries |
| 7 | 3 | -- | collision after 1 safe boundaries |
| 8 | 0 | 3-20 | maximum duration |
| 8 | 1 | 3, 5, 7-20 | maximum duration |
| 8 | 2 | 2-20 | maximum duration |
| 8 | 3 | 2-20 | maximum duration |
| 9 | 0 | -- | maximum duration |
| 9 | 1 | -- | maximum duration |
| 9 | 2 | -- | maximum duration |
| 9 | 3 | -- | maximum duration |
| 10 | 0 | -- | collision after 2 safe boundaries |
| 10 | 1 | -- | collision after 2 safe boundaries |
| 10 | 2 | -- | collision after 2 safe boundaries |
| 10 | 3 | -- | collision after 2 safe boundaries |
| 11 | 0 | -- | collision after 1 safe boundaries |
| 11 | 1 | -- | collision after 1 safe boundaries |
| 11 | 2 | -- | collision after 1 safe boundaries |
| 11 | 3 | -- | maximum duration |
| 12 | 0 | 7-9 | maximum duration |
| 12 | 1 | 6-9 | maximum duration |
| 12 | 2 | 7-8 | maximum duration |
| 12 | 3 | 6-20 | maximum duration |
| 13 | 0 | -- | collision after 4 safe boundaries |
| 13 | 1 | -- | collision after 4 safe boundaries |
| 13 | 2 | -- | collision after 4 safe boundaries |
| 13 | 3 | -- | collision after 6 safe boundaries |
| 14 | 0 | -- | collision after 10 safe boundaries |
| 14 | 1 | -- | maximum duration |
| 14 | 2 | -- | collision after 16 safe boundaries |
| 14 | 3 | -- | collision after 12 safe boundaries |
| 15 | 0 | -- | maximum duration |
| 15 | 1 | -- | maximum duration |
| 15 | 2 | -- | maximum duration |
| 15 | 3 | -- | maximum duration |
| 16 | 0 | -- | collision after 7 safe boundaries |
| 16 | 1 | -- | collision after 5 safe boundaries |
| 16 | 2 | -- | collision after 6 safe boundaries |
| 16 | 3 | -- | collision after 3 safe boundaries |
| 17 | 0 | -- | collision after 1 safe boundaries |
| 17 | 1 | -- | collision after 1 safe boundaries |
| 17 | 2 | -- | collision after 1 safe boundaries |
| 17 | 3 | -- | collision after 2 safe boundaries |
| 18 | 0 | 1-4, 8 | maximum duration |
| 18 | 1 | 1-4, 7-8, 10, 16-20 | maximum duration |
| 18 | 2 | 1-3, 8, 11, 15-20 | maximum duration |
| 18 | 3 | 1-5, 14 | maximum duration |
| 19 | 0 | 6-20 | maximum duration |
| 19 | 1 | 7-20 | maximum duration |
| 19 | 2 | 6-20 | maximum duration |
| 19 | 3 | 7-20 | maximum duration |

No corrective trajectory directly completed the task. At the earliest safe handoff of each successful trajectory, the correction prefix had not contacted the target; persistence changed the robot-scene configuration before the frozen source completed the task.

The windows are non-monotonic: continuing correction can leave and later re-enter source recoverability. A fixed correction duration therefore cannot implement the oracle.

A pass is an existence result for temporal persistence and an oracle handoff. It is not evidence that a learned recoverability critic is ready.
