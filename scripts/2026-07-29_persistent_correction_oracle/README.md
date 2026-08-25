# Persistent-correction oracle

This experiment implements the fixed pilot in
`docs/recoverability_corrective_option/minimal_validation.md`.

It rolls the existing SDP-6k policy for multiple consecutive chunks from the
same three states used by the one-chunk oracle-branch pilot. At every safe
chunk boundary it restores an exact clone and evaluates whether the frozen
source continuation can complete safely.

The pilot is an existence test for the proposed recoverability-gated
corrective option. It is not a deployment method and does not train a model.

## Pilot result

The fixed pilot passes: 32 of 219 safe-prefix temporal boundaries are safe
recoverable handoffs. They occur in three of four correction trajectories on
one of the three states. The earliest successful trajectories require 5, 6,
and 7 correction chunks; no correction trajectory directly completes the
task. Safe windows can disappear and later reappear, so persistence alone is
insufficient and the handoff condition is substantive.

The superseded three-state artifact directory was removed during output
cleanup; the pilot result is summarized above and subsumed by the full result
below.

Disjoint state ranges can be run with `--state-offset` and `--state-limit`.
Use `merge_partitions.py` to validate and merge the resulting directories in
global state order.

## Full result

The unchanged expansion covers 7/20 strict states. Six of the seven
recoverable states have safe handoffs in all four trajectories; all 52
trajectories from the thirteen uncovered states fail. Five recoverable states
require more than one correction chunk, but current SDP coverage remains well
below the 18/20 waypoint-teacher upper bound. See
`outputs/persistent_correction_oracle/full_20/conclusion.md`.
