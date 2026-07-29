# Gated Corrective-Support Selector

This completed experiment keeps the epoch-260 source policy as the default and
uses the SDP-6k policy only when an oracle EEF-circle obstruction gate fires.
It samples 32 corrective chunks, filters them by clearance and target retreat,
and selects the least source-deviating eligible action chunk.

The fixed-state preflight finds eligible corrective samples in 11/20 held-out
pre-collision states. The paired D2 gate nevertheless lowers Task SR from 0.25
to 0.15: it produces zero failure-to-success and two success-to-NCR
transitions. Do not scale or tune this selector on the same gate.

Canonical result:
`outputs/gated_support_selector/report.md`.

Main entry points:

- `fixed_state_preflight.py`
- `paired_rollout_eval.py`
- `selector.py` and `test_selector.py`
