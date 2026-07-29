# Persistent-Correction Oracle Pilot

The existing SDP-6k policy is executed for multiple consecutive closed-loop chunks. Every collision-free chunk boundary is cloned and handed to the frozen source policy.

## Aggregate

- states / trajectories: 3 / 12
- safe-prefix temporal boundaries: 219
- safe recoverable handoffs: 32
- states with a safe recoverable handoff: 1
- trajectories with a safe recoverable handoff: 3
- direct safe corrective successes: 0
- source-continuation outcomes: `{'collision_failure': 39, 'ncr': 142, 'safe_success': 32, 'success_with_collision': 6}`
- pilot decision: **PASS**

## Per state

| state | trajectories | boundaries | trajectories with safe handoff | earliest trajectory / chunk / actions / absolute step |
|---|---:|---:|---:|---|
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_000 | 4 | 80 | 0 | -- |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_001 | 4 | 66 | 3 | 1 / 5 / 40 / 64 |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_002 | 4 | 73 | 0 | -- |

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

No corrective trajectory directly completed the task. At the earliest safe handoff of each successful trajectory, the correction prefix had not contacted the target; persistence changed the robot-scene configuration before the frozen source completed the task.

The windows are non-monotonic: continuing correction can leave and later re-enter source recoverability. A fixed correction duration therefore cannot implement the oracle.

A pass is an existence result for temporal persistence and an oracle handoff. It is not evidence that a learned recoverability critic is ready.
