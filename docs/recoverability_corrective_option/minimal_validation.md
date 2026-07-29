# Persistent-Correction Recoverability Gate

Status: fixed pilot contract, 2026-07-29.

## Question

The existing SDP policy produces locally safe one-chunk corrections, but none
of 96 one-chunk branches on the first three strict D2 states reaches a frozen
source continuation that contacts Can or completes the task. Does repeatedly
executing the same corrective policy create a later state from which the
frozen source can safely complete?

This is the minimum falsification test for a recoverability-gated corrective
option. It does not train a new policy, value function, world model, or
handoff classifier.

## Fixed assets

- states: the first three held-out strict one-chunk-lead D2 collision states
  used by `outputs/oracle_branch_value/pilot_3state/`;
- source: delta-EEF epoch-260;
- correction: the already selected SDP-6k checkpoint;
- correction trajectories per state: 4;
- action chunk: 8 executed actions with closed-loop replanning;
- maximum correction duration: 20 chunks / 160 actions;
- source-continuation horizon: absolute episode step 600;
- candidate handoffs: every collision-free correction chunk boundary;
- one fixed source continuation seed per state, shared across all handoff
  times and correction trajectories.

Correction trajectory `j`, chunk `k` uses a deterministic seed derived from
the saved state seed, `j`, and `k`. The exact formula is recorded in the
manifest.

## Oracle

For each collision-free boundary along a persistent corrective trajectory:

1. clone the exact simulator state and stacked policy observation history;
2. hand control to the frozen source policy;
3. execute until Task success or absolute episode step 600;
4. record Task success, target contact, distractor collision, and completion
   step.

A boundary is a **safe recoverable handoff** only if:

- the corrective prefix has no distractor contact;
- the source continuation completes the task;
- the source continuation has no distractor contact.

The hindsight oracle chooses the earliest safe recoverable boundary for each
trajectory. This oracle is diagnostic only; deployment has no access to future
continuations.

## Gate

- **Pass:** at least one of the three states has a safe recoverable handoff
  under any of its four persistent corrective trajectories.
- **Fail:** zero of three states has a safe recoverable handoff.

If the pilot fails, stop using the existing SDP policy as the corrective
option and train a correction-only temporally extended policy before building
any recoverability critic. If it passes, expand the identical protocol to the
remaining 17 strict states before learning a handoff model.

Do not tune the number of trajectories, correction duration, source horizon,
or seeds after inspecting the pilot.
