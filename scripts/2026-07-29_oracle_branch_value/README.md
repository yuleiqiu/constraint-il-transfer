# Oracle Simulator-Branch Value Diagnostic

This diagnostic tests whether the current source + SDP candidate set contains
task-successful continuations that a temporally informed selector could choose.
Every candidate executes from the same saved simulator state and exact
observation history, then hands control to the frozen source policy for the
remaining episode horizon.

It is an oracle upper bound for deciding whether a learned value/world model is
worth pursuing. It is not a deployable method.

The three-state pilot evaluates 96 SDP branches. Every corrective chunk avoids
immediate physical contact, but none of the full source continuations contacts
Can or succeeds. The explicit waypoint control solves the same states 3/3
without collision. The pilot therefore does not advance to all 20 states.

Do not train a model whose only role is to rank these one-chunk candidates.
See `outputs/oracle_branch_value/pilot_3state/result_report.md`.

Entry points:

- `simulator_branch_eval.py`
- `analyze_results.py`
