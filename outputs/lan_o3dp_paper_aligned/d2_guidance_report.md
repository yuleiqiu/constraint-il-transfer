# LAN-O3DP paper-aligned guidance: D2 result

Date: 2026-08-07

## Protocol

- Environment: `PickPlaceBreadCerealCan` (D2).
- Frozen checkpoint: `model_epoch_50_image_v15_abs_eef_pose_osc_lan40_paper256_hires_continuous_success_0.7.pth`.
- Ten fixed initial simulator states, seed 705, horizon 600.
- Baseline and guided rollouts reuse the same serialized initial state and policy seed.
- Collision is the union of robot/obstacle and carried-target/obstacle contact. Targeted reruns further split robot contact into arm and gripper contact.
- Main paper-scale comparison: `Q=0.04`, `rho in {0.0003, 0.01}`.
- Mechanism follow-up: all four baseline collision states with `rho=0.01`, `Q in {0.08, 0.10, 0.12}`.

## Ten-state result at Q=0.04

| Condition | Task SR | Safe SR | CR | Horizon truncation | Collision -> clear |
|---|---:|---:|---:|---:|---:|
| Baseline | 9/10 | 6/10 | 4/10 | 1/10 | -- |
| rho=0.0003 | 9/10 | 6/10 | 4/10 | 1/10 | 0/4 |
| rho=0.01 | 7/10 | 6/10 | 4/10 | 3/10 | 0/4 |

The completion changes near H600 are not stable across independent reruns, so
they are treated as horizon-boundary noise rather than evidence that guidance
changes task success. The collision labels are stable across the diagnostic
reruns.

## Contact diagnosis

The four baseline collision states are episodes 1, 2, 5, and 6. Targeted
replays show:

- arm/obstacle contact: 0/4;
- gripper/obstacle contact: 4/4;
- carried Can/obstacle contact: 0/4;
- contacted distractor: Cereal in all four states.

Episode 5 is especially diagnostic: its minimum EEF-point to obstacle-cloud
surface clearance is about 0.097 m while the Panda hand still contacts Cereal.
Thus `Q=0.04`, which is based on the Can footprint, does not represent the
gripper footprint.

## Expanded-Q result on the four collision states

| Q | Task success | CR | Gripper CR | Arm CR | Target CR | Mean trigger rate | Collision -> clear |
|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3/4 | 4/4 | 4/4 | 0/4 | 0/4 | 0.0% | -- |
| 0.08 | 3/4 | 4/4 | 4/4 | 0/4 | 0/4 | 52.1% | 0/4 |
| 0.10 | 3/4 | 4/4 | 4/4 | 0/4 | 0/4 | 68.5% | 0/4 |
| 0.12 | 2/4 | 4/4 | 4/4 | 0/4 | 0/4 | 72.7% | 0/4 |

Increasing `Q` activates guidance much more often, but it does not remove any
of the four gripper contacts. At `Q=0.12`, episode 6 changes from a 531-step
success to an H600 truncation while retaining the collision.

## Conclusion

D2 is the informative setting: the visually filtered policy input remains the
same target-plus-goal point cloud, but distractors alter execution and produce
real contacts. Under the evaluated paper-aligned point-to-point guidance,
neither the paper-sized threshold nor a threshold enlarged to cover the hand
produces a collision-to-clear transition. The immediate limitation is not
that guidance fails to trigger; it is that the cost models the EEF trajectory
as points and does not model the swept gripper geometry or a temporally
persistent bypass route.

The next mechanism-level test should therefore be a swept gripper-geometry
cost (reported explicitly as a departure from the paper), rather than a larger
blind `rho`/`Q` sweep.

## Artifacts

- Fixed state bank: `d2_state_bank_seed705_n10/initial_state_bank.pkl`
- Ten-state merged metrics: `d2_combined_collision_seed705_n10_merged/`
- Contact-category reruns: `d2_contact_category_seed705_ep1_2_gpu0/` and
  `d2_contact_category_seed705_ep5_6_gpu1/`
- Expanded-Q reruns: `d2_q_expanded_seed705_ep1_2_gpu0/` and
  `d2_q_expanded_seed705_ep5_6_gpu1/`
