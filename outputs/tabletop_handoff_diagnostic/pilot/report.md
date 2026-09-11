# Tabletop oracle-route diagnostic

Selected transport states; repeated seeds are replay audits, not independent trials.

| Condition | Runs | Task SR | Safe SR | CR | NCR | Four-way counts: safe / success+contact / collision failure / NCR |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0 / 0 / 1 / 0 |
| guided | 1 | 0.000 | 0.000 | 0.000 | 1.000 | 0 / 0 / 0 / 1 |
| short | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1 / 0 / 0 / 0 |
| long | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1 / 0 / 0 / 0 |
| teacher | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1 / 0 / 0 / 0 |

See analysis.json for paired transitions, individual-layout repeat outcomes, boundary state / observation differences and first-chunk reproduction.

These are oracle-route feasibility diagnostics, not learned-method performance or evidence against every local guidance method.
