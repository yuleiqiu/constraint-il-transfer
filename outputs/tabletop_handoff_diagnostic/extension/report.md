# Tabletop oracle-route diagnostic

Selected transport states; repeated seeds are replay audits, not independent trials.

| Condition | Runs | Task SR | Safe SR | CR | NCR | Four-way counts: safe / success+contact / collision failure / NCR |
|---|---:|---:|---:|---:|---:|---|
| baseline | 3 | 0.000 | 0.000 | 1.000 | 0.000 | 0 / 0 / 3 / 0 |
| guided | 3 | 1.000 | 1.000 | 0.000 | 0.000 | 3 / 0 / 0 / 0 |
| short | 3 | 0.000 | 0.000 | 1.000 | 0.000 | 0 / 0 / 3 / 0 |
| long | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 0 / 0 / 0 / 3 |
| near_goal | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 0 / 0 / 0 / 3 |
| teacher | 3 | 1.000 | 1.000 | 0.000 | 0.000 | 3 / 0 / 0 / 0 |

See analysis.json for paired transitions, individual-layout repeat outcomes, boundary state / observation differences and first-chunk reproduction.

These are oracle-route feasibility diagnostics, not learned-method performance or evidence against every local guidance method.
