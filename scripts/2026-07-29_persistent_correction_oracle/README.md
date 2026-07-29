# Persistent-correction oracle

This experiment implements the fixed pilot in
`docs/recoverability_corrective_option/minimal_validation.md`.

It rolls the existing SDP-6k policy for multiple consecutive chunks from the
same three states used by the one-chunk oracle-branch pilot. At every safe
chunk boundary it restores an exact clone and evaluates whether the frozen
source continuation can complete safely.

The pilot is an existence test for the proposed recoverability-gated
corrective option. It is not a deployment method and does not train a model.

## Result

The fixed pilot passes: 32 of 219 safe-prefix temporal boundaries are safe
recoverable handoffs. They occur in three of four correction trajectories on
one of the three states. The earliest successful trajectories require 5, 6,
and 7 correction chunks; no correction trajectory directly completes the
task. Safe windows can disappear and later reappear, so persistence alone is
insufficient and the handoff condition is substantive.

See `outputs/persistent_correction_oracle/pilot_3state/report.md`.
