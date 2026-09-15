# Full50 comparison conclusion

Completed100 rollouts; each has a verified actual-execution MP4.
Frozen compact-Bezier planner versus source on the original50-layout bank.

| Condition | Task | Safe | CR | NCR | Safe / success+contact / collision failure / NCR |
|---|---:|---:|---:|---:|---|
| baseline | 46/50 | 39/50 | 9/50 | 2/50 | 39 / 7 / 2 / 2 |
| planner | 49/50 | 49/50 | 0/50 | 1/50 | 49 / 0 / 0 / 1 |

Paired transitions:
- success: 3 rescues / 0 regressions.
- safe_success: 10 rescues / 0 regressions.

See summary.json for episode IDs and video_index.md for manual review.
This evaluates the complete development bank, not universal safety or held-out generalization.
The planner uses oracle geometry and known goal position. No parameter tuning within this matrix.
Initial observations are checked within each pair; archived baseline prefixes and video artifacts are verified.
No repeated-outcome claim is made for all50 layouts. Earlier0/25/28 repeats remain separate.

## Review2026-09-14

Task rescues:14,34,43. Safe rescues:8,9,14,18,25,28,34,37,43,49.
No task or safe regressions. Planner27 is the sole failure: no obstacle contact,
H600, with one completed bypass and handoff. Source27 also fails, with contact.
This identifies the remaining case, not its final failure mechanism.

All100 videos verified; execution plus final checks took1782.9 seconds.
