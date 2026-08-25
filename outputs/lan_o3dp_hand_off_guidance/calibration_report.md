# LAN-O3DP hand-off guidance calibration

Date: 2026-08-12

## Mechanism smoke

The four previously diagnosed D2 hand/Cereal-contact states were replayed with
only `gripper0_right_hand_collision` disabled. All finger mesh and pad contact
masks remained unchanged on every rollout.

- Task success changed from 3/4 under the retained full-physics replay to 4/4.
- Collision changed from 4/4 hand contact to 1/4 residual finger contact.
- Episodes 1, 2, and 5 became collision-free successes.
- Episode 6 retained `gripper0_right_finger1_collision` with Cereal and still
  completed the task.

## Q selection

On the residual episode 6, near-zero-scale probes found:

| Q | epsilon | First active step | First finger-contact step | Trigger rate |
|---:|---:|---:|---:|---:|
| 0.04 | 0.01 | 24 | 92 | 38.3% |
| 0.06 | 0.015 | 16 | 92 | 50.1% |
| 0.08 | 0.02 | 0 | 92 | 58.7% |

`Q=0.04 m` is the smallest candidate that activates well before contact. The
Huber transition width is frozen to `epsilon=0.25Q=0.01 m`.

## Rho selection

Ten fixed D2 calibration states were evaluated with hand collision disabled,
H600, and the frozen XY Huberized-hinge cost.

| rho | Task | Safe | Collision | Collision -> clear | Clear -> collision |
|---:|---:|---:|---:|---:|---:|
| baseline | 10/10 | 8/10 | 2/10 | -- | -- |
| 0.001 | 10/10 | 8/10 | 2/10 | 0 | 0 |
| 0.003 | 10/10 | 7/10 | 3/10 | 0 | 1 |
| 0.01 | 10/10 | 9/10 | 1/10 | 1 | 0 |
| 0.03 | 10/10 | 8/10 | 2/10 | 0 | 0 |

Thus `rho=0.01` is frozen. A five-state D1 preservation check at the selected
parameters kept Task/Safe success at 4/5, CR at 0/5, and had zero paired task
transitions. Its mean guidance trigger rate was 7.7%.

The calibration states are excluded from the formal seed-812 evaluation.
