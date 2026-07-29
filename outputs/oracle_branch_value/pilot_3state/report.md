# Oracle Simulator-Branch Value Upper Bound

Each branch executes one source or SDP action chunk from the same saved state, then hands control to the frozen source policy for the remaining episode horizon.

## Aggregate

- states: 3
- source outcomes: `{'collision_failure': 3}`
- oracle over all candidates: `{'ncr': 1, 'collision_failure': 2}`
- oracle over source + geometry-eligible candidates: `{'ncr': 1, 'collision_failure': 2}`
- states with any corrective success / safe success: 0 / 0
- states with eligible corrective success / safe success: 0 / 0
- oracle-any failure→success / success→failure: 0 / 0
- oracle-eligible failure→success / success→failure: 0 / 0

## Per state

| state | source | local selector | oracle any | oracle eligible | corrective success / safe | eligible |
|---|---|---|---|---|---:|---:|
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_000 | collision_failure | ncr (#22) | ncr (#2) | ncr (#2) | 0 / 0 | 30 |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_001 | collision_failure | collision_failure (#-1) | collision_failure (#16) | collision_failure (#-1) | 0 / 0 | 0 |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_002 | collision_failure | collision_failure (#-1) | collision_failure (#19) | collision_failure (#-1) | 0 / 0 | 0 |
