# Control Timing Diagnosis

Dataset:

```text
third_party/robomimic/datasets/can/yq/image_v15.hdf5
```

Script:

```text
scripts/diagnose_control_timing.py
```

Raw output:

```text
outputs/control_timing/control_timing.json
```

## Results

| Quantity | Value |
|---|---:|
| Runtime control frequency | `20 Hz` |
| Duration of one `env.step(action)` | `0.05 s` |
| MuJoCo / robosuite model timestep | `0.002 s` |
| Physics substeps per action | `25` |
| Dataset recorded EEF poses per action | `1` |
| Dataset EEF pose interval | `0.05 s` |
| H=16 action chunk duration | `0.8 s` |

## Interpretation

Externally, the policy and dataset operate at the control timestep:

```text
1 action -> env.step(action) -> 1 recorded next EEF pose
```

The OSC controller holds one action for:

```text
1 / 20 Hz = 0.05 s
```

Inside that one control step, robosuite advances MuJoCo at:

```text
0.002 s per physics substep
```

so there are:

```text
0.05 / 0.002 = 25 physics substeps
```

The dataset and the forward model do not record / predict those 25 intermediate EEF poses. They use one EEF pose per control step:

```text
actions[t] -> next_obs/robot0_eef_pos[t]
```

Therefore, for horizon `H=16`:

```text
16 actions -> 16 future EEF poses -> 0.8 s
```
