# Rho trajectory-separation sweep

Status: closed negative diagnostic. The large per-rollout sweep directories
were removed during the 2026-08-18 output cleanup; this compact conclusion is
retained because it is referenced by the research log.

Across four fixed layouts, increasing `rho` did not produce an early,
monotonic, task-preserving bypass. In episode 0 the guidance first activates at
step 48 and collision occurs at step 53; even `rho=10` changes the executed
trajectory by only 0.01 mm before collision. In episode 4, `rho=10` produces
1.19 cm separation before collision, but the finger collision and task outcome
remain unchanged. The collision-clear episode 35 never reaches 1 cm separation
at any `rho<=1`.

Task success remains 2/4 for baseline and every main-sweep value. Collision
labels change non-monotonically. Position clipping is not a complete
explanation: episode 0 at `rho=0.03` reaches 8.76 cm maximum late separation
with no position clipping. The supported interpretation is that small changes
are amplified only after contact or task failure, rather than forming a useful
pre-contact avoidance route.

Decision: do not select a larger `rho` or run a formal matrix for the same XY
point cost.
