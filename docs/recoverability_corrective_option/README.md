# Recoverability-Gated Corrective Option

Status: method hypothesis supported by a three-state oracle pilot,
2026-07-29.

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

## Minimum oracle evidence

The fixed three-state pilot executed the existing SDP-6k policy persistently
for at most 20 closed-loop chunks and tested the frozen source at every safe
chunk boundary.

- 219 safe-prefix boundaries were tested.
- 32 were safe recoverable handoffs.
- The existence gate passed on one of three states.
- Three of four correction trajectories on that state contained a safe
  handoff window.
- The earliest safe windows began after 5, 6, and 7 correction chunks
  (40, 48, and 56 actions).
- No correction trajectory directly completed the task, and none had contacted
  the target at its earliest successful handoff.
- Recoverability was non-monotonic: trajectories could enter, leave, and later
  re-enter a safe handoff window.

Thus temporal persistence creates behavior that the rejected one-chunk
candidate set did not contain, and a handoff decision is genuinely necessary.
The result is still an existence diagnostic on selected D2 states, not a
learned method or a rollout-level success claim.

## Next gate

Apply the identical oracle protocol to the remaining 17 strict states before
training a recoverability estimator. That expansion should measure state
coverage, window width, correction collision rate, and the amount of
correction required. If coverage remains sparse, replace the mixed-policy
SDP checkpoint with a correction-only temporally extended policy before
learning the handoff model.

Fixed pilot contract:
`docs/recoverability_corrective_option/minimal_validation.md`. Canonical
result: `outputs/persistent_correction_oracle/pilot_3state/report.md`.
