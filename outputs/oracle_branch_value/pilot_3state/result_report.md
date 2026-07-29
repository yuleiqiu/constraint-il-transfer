# Oracle Branch-Value Pilot

Status: stopped at the preregistered three-state pilot gate.

## Question

Does the existing SDP policy produce at least one one-chunk corrective candidate whose full frozen-source continuation succeeds, so that a learned value/world model would have a useful branch to select?

## Protocol

- first three held-out one-chunk-lead collision states;
- one source plus 32 SDP chunks per state;
- exact saved simulator state and policy observation history restored for every branch;
- execute the candidate's eight actions, then run the frozen source until the original 400-step episode horizon;
- identical source continuation noise seeds across branches;
- record full task success, contacts, collision, staged rewards, and 8/32/64/128-step checkpoints.

The pilot advances to all 20 states only if at least one corrective branch completes the task. This is an oracle existence gate, not a statistical performance estimate.

## Result

| quantity | result |
|---|---:|
| saved states | 3 |
| source / corrective branches | 3 / 96 |
| corrective Task successes | 0 / 96 |
| corrective Can-contact branches | 0 / 96 |
| corrective immediate collision-free branches | 96 / 96 |
| corrective full collision-free branches | 32 / 96 |
| geometry-eligible branches | 30 / 96 |
| action clipping | 0 |

Corrective outcomes: `{'ncr': 32, 'collision_failure': 64}`.
Even a hindsight oracle over all 32 candidates obtains `{'ncr': 1, 'collision_failure': 2}`; restricting it to the source plus geometry-eligible candidates obtains `{'ncr': 1, 'collision_failure': 2}`.

## Per-state structure

| state | eligible | immediate safe | Can contact | success | local | oracle any |
|---|---:|---:|---:|---:|---|---|
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_000 | 30 | 32 | 0 | 0 | ncr | ncr |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_001 | 0 | 32 | 0 | 0 | collision_failure | collision_failure |
| one_chunk_lead_states:PickPlaceBreadCerealCan_preentry_002 | 0 | 32 | 0 | 0 | collision_failure | collision_failure |

The first state demonstrates the strongest local/global mismatch: all 32 corrective chunks avoid immediate contact and 30 satisfy the geometry filter, but all full continuations are NCR. In the other two states, all 64 corrective continuations are collision failures and none contacts Can.

## Explicit waypoint control

The existing discrete-yaw waypoint teacher was rerun on the same three simulator states. It obtains 3/3 Task success, 3/3 Can contact, and 0/3 distractor collision, using 121--144 controlled steps before handoff. Thus these states are solvable; the SDP one-chunk candidate set is the failed component.

## Decision

Do not expand this diagnostic to 20 states and do not train a world model whose only role is to rank the current one-chunk SDP candidates. On the pilot, there is no successful future for such a model to select.

This does not reject long-horizon model-based planning that generates new multi-step behavior. It rejects the narrower and cheaper hypothesis that selection alone is the remaining problem. The positive waypoint control indicates that the next useful model should represent an explicit multi-step avoidance/grasp route or residual expert, with the source frozen and a defined handoff.

## Validation

- maximum source-vs-captured action reproduction difference: 3.105e-05;
- action clips: 0;
- mean full-branch runtime: 6.76 s.
