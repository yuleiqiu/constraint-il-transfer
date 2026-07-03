# Delta EEF to OSC Adapter Report

**Date**: 2026-06-29

**Dataset**: `third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5`

**Question**: Can we convert a predicted real EEF displacement back into the
original OSC command, so that a `delta_eef_action` policy can still execute in
the existing robosuite `OSC_POSE` environment?

## TL;DR

**No, not reliably enough for execution.** The mapping is partially predictable:
linear and neural adapters recover the coarse direction and scale of the
original OSC command. However, they fail open-loop replay with centimeter-level
drift and occasional catastrophic divergence. Therefore, `real delta_EEF ->
OSC command` is not a usable bridge for Route B under the current robosuite OSC
interface.

This is not a claim of mathematical impossibility. It is an empirical
deployment conclusion: the conversion is not accurate enough to serve as an
executable action interface for imitation policy rollout or trajectory
guidance.

## Setup

The dataset contains both sides of the mapping:

```text
actions[:, :3]            = original normalized OSC position command
delta_eef_action[:, :3]   = actual EEF displacement / 0.05
obs/*                     = robot state before action
```

We tested two adapter families:

1. **Closed-form adapters**
   - scalar scale
   - diagonal scale
   - full linear map

2. **State-conditioned MLP**
   - input: `robot0_eef_pos`, `robot0_eef_quat`, `robot0_joint_pos`,
     `robot0_joint_vel`, `delta_eef_action[:3]`
   - output: `actions[:3]`

Rotation and gripper dimensions were copied directly from `delta_eef_action`,
because those fields already match the original action semantics in this
dataset.

## Phase 1: Offline Fitting

The closed-form adapters were fit with:

```bash
uv run python docs/route_b_validation/fit_delta_eef_to_osc_adapter.py
```

Outputs:

```text
outputs/route_b_validation/delta_eef_to_osc_adapter/adapter_params.json
outputs/route_b_validation/delta_eef_to_osc_adapter/metrics.json
outputs/route_b_validation/delta_eef_to_osc_adapter/summary.md
```

The fitted scale is close to the expected inverse OSC tracking ratio:

```text
scalar scale = 3.734
diagonal scale = [3.824, 3.789, 3.513]
```

Valid split metrics:

| adapter | mse_xyz | mae_xyz | cosine_median | clip_any_xyz |
|---|---:|---:|---:|---:|
| scalar scale | 0.002882 | 0.02938 | 0.9935 | 0.0003 |
| diagonal scale | 0.002845 | 0.02901 | 0.9937 | 0.0003 |
| full linear | 0.002516 | 0.02922 | 0.9924 | 0.0003 |

Interpretation: `delta_eef_action` predicts the rough OSC command direction
and scale, but offline metrics alone do not test whether the conversion is
stable when executed through the controller.

## Phase 2: Open-Loop Replay

The adapters were replayed through the original OSC environment:

```bash
MUJOCO_GL=egl uv run python docs/route_b_validation/replay_delta_eef_to_osc_adapter.py --no-mlp
uv run python docs/route_b_validation/plot_delta_eef_to_osc_adapter_replay.py
```

The replay command for adapter plans is:

```text
delta_eef_action[t] -> adapter -> env.step(osc_command)
```

Results on the first 5 `valid` demos:

| plan | mean target err cm | end orig err cm |
|---|---:|---:|
| Plan A: original OSC action | 0.407 | 0.346 |
| Plan B-1: raw delta EEF as OSC | 21.220 | 37.893 |
| scalar adapter | 3.494 | 8.840 |
| diagonal adapter | 3.254 | 8.848 |
| full linear adapter | 2.427 | 4.864 |

The full linear adapter substantially improves over raw `delta_eef_action`, but
it is still an order of magnitude worse than Plan A. The worst validation demo
still diverges to 17.7 cm end error.

## Phase 3: State-Conditioned MLP

The MLP adapter was trained with:

```bash
uv run python docs/route_b_validation/train_delta_eef_to_osc_mlp_adapter.py
```

Outputs:

```text
outputs/route_b_validation/delta_eef_to_osc_adapter/mlp_adapter.pth
outputs/route_b_validation/delta_eef_to_osc_adapter/mlp_adapter_metrics.json
outputs/route_b_validation/delta_eef_to_osc_adapter/mlp_adapter_summary.md
```

Offline valid metrics improved:

| adapter | mse_xyz | mae_xyz | cosine_median | clip_any_xyz |
|---|---:|---:|---:|---:|
| full linear | 0.002516 | 0.02922 | 0.9924 | 0.0003 |
| state-conditioned MLP | 0.000785 | 0.01884 | 0.9960 | 0.0000 |

But open-loop replay did not pass:

| plan | mean target err cm | end orig err cm |
|---|---:|---:|
| Plan A: original OSC action | 0.407 | 0.346 |
| full linear adapter | 2.427 | 4.864 |
| state-conditioned MLP | 2.301 | 7.002 |

The MLP slightly improves mean target error but worsens end-of-trajectory error.
One validation demo (`demo_140`) diverges to 28.9 cm end error. The MLP was
evaluated using the **current replay environment observation**, not the dataset
observation, so this failure reflects realistic state drift: once the replay
trajectory leaves the dataset manifold, the adapter sees off-distribution robot
states and its correction can compound error.

## Conclusion

The adapter experiments answer the conversion question directly:

```text
real delta_EEF -> OSC command
```

This mapping is learnable in an offline regression sense, but it is not stable
enough under open-loop execution. Even the best adapters remain far from the
original OSC replay upper bound, and the state-conditioned MLP shows that better
offline prediction does not guarantee better controller-level replay.

For Route B, this means:

```text
delta_eef policy -> inverse adapter -> existing OSC controller
```

is not a reliable solution. A future EEF-supervised policy should instead use
an environment/control interface that natively accepts EEF pose, EEF delta, or
Cartesian trajectory commands, and that interface must pass open-loop replay
before training.

## Reproduction

Run all phases from the repo root:

```bash
uv run python docs/route_b_validation/fit_delta_eef_to_osc_adapter.py
MUJOCO_GL=egl uv run python docs/route_b_validation/replay_delta_eef_to_osc_adapter.py --no-mlp
uv run python docs/route_b_validation/train_delta_eef_to_osc_mlp_adapter.py
MUJOCO_GL=egl uv run python docs/route_b_validation/replay_delta_eef_to_osc_adapter.py
uv run python docs/route_b_validation/plot_delta_eef_to_osc_adapter_replay.py
```

Final outputs are under:

```text
outputs/route_b_validation/delta_eef_to_osc_adapter/
```
