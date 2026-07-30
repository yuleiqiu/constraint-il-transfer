# Recoverability-Gated Corrective Option

Status: method hypothesis supported; current SDP generator coverage bounded on
all 20 strict states, 2026-07-30.

## Motivation

The frozen source policy contains reliable task behavior but lacks robust
avoidance support in clutter. Full-policy correction training adds local safe
actions, yet can disrupt the source task modes. One-chunk corrective sampling
also fails because a locally safe action does not necessarily leave the system
in a state from which the source can finish.

The proposed abstraction is therefore not another full task policy and not a
one-step action selector. It is a **temporally persistent corrective option**
whose terminal condition is return to the frozen source policy's safe
recoverability set.

## Candidate method

- Freeze the source policy \(\pi_0\).
- Invoke a separate correction policy \(\pi_c\) when predicted source
  execution is obstructed.
- Latch the corrective mode across action chunks instead of re-deciding from
  scratch every eight actions.
- Estimate source recoverability
  \(V_0(o)=P(\text{safe task completion by }\pi_0\mid o)\).
- Hand control back only when safety constraints hold and \(V_0(o)\) crosses a
  calibrated threshold; use hysteresis to prevent rapid switching.

The correction policy should learn obstacle-relative recovery motion rather
than the complete grasp-and-place task. Simulator source continuations provide
labels for \(V_0\); the value model is a handoff model, not a general-purpose
world model.

This boundary intentionally excludes an LLM/VLM phase parser, a general
dynamics world model, and task-specific named checkpoints such as "lift 5 cm."

## Oracle evidence

The fixed pilot and its unchanged 17-state expansion executed the existing
SDP-6k policy persistently for at most 20 closed-loop chunks and tested the
frozen source at every safe chunk boundary.

- 7/20 states contain a safe recoverable handoff.
- Within those states, 27/28 correction trajectories contain a safe handoff;
  the remaining thirteen states have 0/52.
- Five of seven recoverable states require more than one correction chunk.
- The median earliest state-level handoff is five chunks.
- No correction trajectory directly completed the task.
- Recoverability was non-monotonic: trajectories could enter, leave, and later
  re-enter a safe handoff window.

Thus temporal persistence creates behavior that the rejected one-chunk
candidate set did not contain, and a handoff decision is genuinely necessary.
The result is still an oracle existence diagnostic, not a learned method or a
rollout-level success claim.

## Next gate

The waypoint teacher solves 18/20 of the same states, while persistent SDP
covers only 7/20 even with an oracle handoff. Replace the mixed-policy SDP
checkpoint with a correction-only temporally extended option trained from
the existing verified intervention segments, then repeat the identical
20-state oracle gate. Collect more segments only if the stored sequences
cannot support that interface. Do not train the handoff estimator until the
behavior generator materially expands coverage.

Fixed pilot contract:
`docs/recoverability_corrective_option/minimal_validation.md`. Canonical
result: `outputs/persistent_correction_oracle/pilot_3state/report.md`.
Full result:
`outputs/persistent_correction_oracle/full_20/conclusion.md`.
