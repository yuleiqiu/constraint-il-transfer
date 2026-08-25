# Tabletop counterfactual rescue evaluation

Paired triads: 25 (five layouts x five diffusion-noise seeds).
Clean successes: 17/25.
Eligible clean-success / obstacle-collision-failure pairs: 11.
Guided safe task-completing rescues: 4/11 = 0.364.

## Per-scene rescue counts

| Scene | Pairs | Clean success | Eligible | Safe rescues | Rescue rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 0 | 0 | n/a |
| 4 | 5 | 5 | 5 | 4 | 0.80 |
| 5 | 5 | 0 | 0 | 0 | n/a |
| 7 | 5 | 4 | 4 | 0 | 0.00 |
| 8 | 5 | 3 | 2 | 0 | 0.00 |

## Eligibility-scoped guided outcomes

```json
{
  "safe_success": 4,
  "collision_free_non_completion": 6,
  "collision_failure": 1
}
```

## Collision-pair audit on eligible pairs

```json
{
  "obstacle_baseline": {
    "carried_can:Milk": 11,
    "carried_can:MilkSupport": 1,
    "gripper:Milk": 5,
    "gripper_finger:Milk": 4,
    "gripper_hand:Milk": 4
  },
  "obstacle_guided": {
    "carried_can:Milk": 1,
    "gripper:Milk": 1,
    "gripper_hand:Milk": 1
  }
}
```

The EEF is a site rather than a collision geom. EEF-versus-Milk is therefore reported as an XY footprint-overlap proxy; physical contacts are separated into gripper hand, gripper finger, arm link, and carried Can.
