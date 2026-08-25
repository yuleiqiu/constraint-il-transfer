# LAN-O3DP hand-off guidance: formal 2x2 result

## Result

Disabling only the Panda hand collision shell is an effective physics relaxation in the hard environments, but it does **not** unlock a material guidance benefit. In D2, hand-off changes baseline Task SR from 25/50 to 31/50 while Safe SR stays 17/50; the extra completions therefore remain collision-contaminated. In D3 it changes baseline Task/Safe SR from 36/50 and 17/50 to 39/50 and 24/50.

With hand collision disabled, guidance changes D2 Task/Safe SR from 31/50 and 17/50 to 31/50 and 18/50, and D3 from 39/50 and 24/50 to 40/50 and 25/50. Every paired guidance comparison has exact McNemar `p >= 0.5`; the observed net changes are one episode at most.

The contact split explains the limit. Under full physics, guidance does not change hand CR at all (D2 12/50 to 12/50; D3 19/50 to 19/50). Under hand-off, residual finger CR changes from 16/50 to 18/50 in D2 and stays 17/50 in D3. Target-object CR falls, but the XY point-EEF cost still does not model the swept finger geometry or produce a persistent bypass route.

Decision: retain hand-off as a useful mechanism diagnostic, not as evidence that the current 3-D guidance path succeeds. Do not scale a larger blind `rho` / `Q` sweep of this same XY point cost. Also treat hand-off safety numbers as modified-physics results: the hand shell cannot make any physical contact, including with the target or table.

## Protocol and integrity

Frozen policy: epoch-50 corrected high-resolution LAN-O3DP sample checkpoint (`...paper256_hires_continuous_success_0.7.pth`). Its observation is `agent_pos` plus a 256-point world-frame point cloud containing only Can and VisualCan; Bread, Cereal, and Milk are excluded from the policy input and supplied only to guidance through the oracle obstacle cloud.

Frozen protocol: seed 812, 50 paired states per environment, H600, XY Huberized hinge, `Q=0.04 m`, `epsilon=0.01 m`, `rho=0.01`.
The seed-705 D2 calibration bank is excluded. All 800 condition rollouts passed state-hash, policy-seed, per-row parameter, and collision-mask checks. Only `gripper0_right_hand_collision` is disabled in hand-off; all finger mesh and pad masks are byte-for-byte unchanged from full physics.

## Aggregate outcomes

| Env | Physics | Policy | Task SR | Safe SR | CR | NCR | Hand CR | Finger CR | Arm CR | Target CR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | full_physics | baseline | 43/50 | 43/50 | 0/50 | 7/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| D0 | full_physics | guided | 43/50 | 43/50 | 0/50 | 7/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| D0 | hand_off | baseline | 44/50 | 44/50 | 0/50 | 6/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| D0 | hand_off | guided | 44/50 | 44/50 | 0/50 | 6/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| D1 | full_physics | baseline | 42/50 | 41/50 | 1/50 | 8/50 | 0/50 | 1/50 | 0/50 | 1/50 |
| D1 | full_physics | guided | 42/50 | 41/50 | 2/50 | 7/50 | 0/50 | 2/50 | 0/50 | 1/50 |
| D1 | hand_off | baseline | 41/50 | 40/50 | 1/50 | 9/50 | 0/50 | 1/50 | 0/50 | 1/50 |
| D1 | hand_off | guided | 41/50 | 40/50 | 1/50 | 9/50 | 0/50 | 1/50 | 0/50 | 1/50 |
| D2 | full_physics | baseline | 25/50 | 17/50 | 22/50 | 11/50 | 12/50 | 14/50 | 4/50 | 12/50 |
| D2 | full_physics | guided | 24/50 | 17/50 | 20/50 | 13/50 | 12/50 | 15/50 | 3/50 | 9/50 |
| D2 | hand_off | baseline | 31/50 | 17/50 | 19/50 | 14/50 | 0/50 | 16/50 | 3/50 | 6/50 |
| D2 | hand_off | guided | 31/50 | 18/50 | 19/50 | 13/50 | 0/50 | 18/50 | 3/50 | 6/50 |
| D3 | full_physics | baseline | 36/50 | 17/50 | 29/50 | 4/50 | 19/50 | 18/50 | 1/50 | 13/50 |
| D3 | full_physics | guided | 36/50 | 18/50 | 28/50 | 4/50 | 19/50 | 18/50 | 2/50 | 9/50 |
| D3 | hand_off | baseline | 39/50 | 24/50 | 22/50 | 4/50 | 0/50 | 17/50 | 1/50 | 12/50 |
| D3 | hand_off | guided | 40/50 | 25/50 | 20/50 | 5/50 | 0/50 | 17/50 | 2/50 | 9/50 |

## Hard-environment four-way partition

| Env | Physics | Policy | Safe success | Success + collision | Collision failure | NCR |
|---|---|---|---:|---:|---:|---:|
| D2 | full_physics | baseline | 17 | 8 | 14 | 11 |
| D2 | full_physics | guided | 17 | 7 | 13 | 13 |
| D2 | hand_off | baseline | 17 | 14 | 5 | 14 |
| D2 | hand_off | guided | 18 | 13 | 6 | 13 |
| D3 | full_physics | baseline | 17 | 19 | 10 | 4 |
| D3 | full_physics | guided | 18 | 18 | 10 | 4 |
| D3 | hand_off | baseline | 24 | 15 | 7 | 4 |
| D3 | hand_off | guided | 25 | 15 | 5 | 5 |

## Guidance mechanism activity

| Env | Physics | Active episodes | Mean trigger rate | Max update norm | Active before collision | Missing-cloud chunks / episodes |
|---|---|---:|---:|---:|---:|---:|
| D1 | full_physics | 20/50 | 5.3% | 8.188e-04 | 2 | 24 / 3 |
| D1 | hand_off | 20/50 | 5.2% | 6.511e-04 | 1 | 25 / 4 |
| D2 | full_physics | 41/50 | 21.5% | 4.706e-03 | 14 | 0 / 0 |
| D2 | hand_off | 41/50 | 20.9% | 1.923e-03 | 16 | 0 / 0 |
| D3 | full_physics | 49/50 | 29.4% | 3.299e-03 | 22 | 0 / 0 |
| D3 | hand_off | 49/50 | 27.1% | 3.144e-03 | 17 | 0 / 0 |

## Paired transitions

### D0

- `guidance_with_full_physics`: task 0->1 / 1->0 = `0/0`; collision 1->0 / 0->1 = `0/0`; safe 0->1 / 1->0 = `0/0`; exact p(task/safe/CR) = `1/1/1`.
- `guidance_with_hand_off`: task 0->1 / 1->0 = `0/0`; collision 1->0 / 0->1 = `0/0`; safe 0->1 / 1->0 = `0/0`; exact p(task/safe/CR) = `1/1/1`.
- `hand_off_with_baseline`: task 0->1 / 1->0 = `1/0`; collision 1->0 / 0->1 = `0/0`; safe 0->1 / 1->0 = `1/0`; exact p(task/safe/CR) = `1/1/1`.
- `hand_off_with_guided`: task 0->1 / 1->0 = `1/0`; collision 1->0 / 0->1 = `0/0`; safe 0->1 / 1->0 = `1/0`; exact p(task/safe/CR) = `1/1/1`.

### D1

- `guidance_with_full_physics`: task 0->1 / 1->0 = `0/0`; collision 1->0 / 0->1 = `0/1`; safe 0->1 / 1->0 = `0/0`; exact p(task/safe/CR) = `1/1/1`.
- `guidance_with_hand_off`: task 0->1 / 1->0 = `0/0`; collision 1->0 / 0->1 = `0/0`; safe 0->1 / 1->0 = `0/0`; exact p(task/safe/CR) = `1/1/1`.
- `hand_off_with_baseline`: task 0->1 / 1->0 = `0/1`; collision 1->0 / 0->1 = `0/0`; safe 0->1 / 1->0 = `0/1`; exact p(task/safe/CR) = `1/1/1`.
- `hand_off_with_guided`: task 0->1 / 1->0 = `0/1`; collision 1->0 / 0->1 = `1/0`; safe 0->1 / 1->0 = `0/1`; exact p(task/safe/CR) = `1/1/1`.

### D2

- `guidance_with_full_physics`: task 0->1 / 1->0 = `0/1`; collision 1->0 / 0->1 = `2/0`; safe 0->1 / 1->0 = `1/1`; exact p(task/safe/CR) = `1/1/0.5`.
- `guidance_with_hand_off`: task 0->1 / 1->0 = `0/0`; collision 1->0 / 0->1 = `1/1`; safe 0->1 / 1->0 = `1/0`; exact p(task/safe/CR) = `1/1/1`.
- `hand_off_with_baseline`: task 0->1 / 1->0 = `7/1`; collision 1->0 / 0->1 = `4/1`; safe 0->1 / 1->0 = `2/2`; exact p(task/safe/CR) = `0.07031/1/0.375`.
- `hand_off_with_guided`: task 0->1 / 1->0 = `7/0`; collision 1->0 / 0->1 = `3/2`; safe 0->1 / 1->0 = `2/1`; exact p(task/safe/CR) = `0.01562/1/1`.

### D3

- `guidance_with_full_physics`: task 0->1 / 1->0 = `0/0`; collision 1->0 / 0->1 = `3/2`; safe 0->1 / 1->0 = `2/1`; exact p(task/safe/CR) = `1/1/1`.
- `guidance_with_hand_off`: task 0->1 / 1->0 = `1/0`; collision 1->0 / 0->1 = `3/1`; safe 0->1 / 1->0 = `2/1`; exact p(task/safe/CR) = `1/1/0.625`.
- `hand_off_with_baseline`: task 0->1 / 1->0 = `3/0`; collision 1->0 / 0->1 = `7/0`; safe 0->1 / 1->0 = `7/0`; exact p(task/safe/CR) = `0.25/0.01562/0.01562`.
- `hand_off_with_guided`: task 0->1 / 1->0 = `4/0`; collision 1->0 / 0->1 = `9/1`; safe 0->1 / 1->0 = `8/1`; exact p(task/safe/CR) = `0.125/0.03906/0.02148`.
