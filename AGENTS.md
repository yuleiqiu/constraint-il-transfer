# Project Context

## 1. Goal

Single-to-multi-object diffusion policy transfer. Decompose failure modes into two orthogonal dimensions:

- **Part A (Δvis)**: Visual ambiguity — "which object is the target?"
- **Part B (Δgeo)**: Physical obstruction — "arm trajectory blocked by new objects"

## 2. Current State

- Masked-image policy (π_mask) trained on PickPlaceCan (oracle mask input)
- Large-scale experiment (600 rollouts × 4 environments) completed: **inference-time guided obstacle avoidance does not improve success rate**
- **Root cause confirmed**: `action → trajectory` mapping (`cumsum(action * 0.05)`) has 3-4cm RMSE vs. OSC PD-controller actual dynamics, comparable to obstacle radius (3-5cm), making cost guidance unreliable
- **Next step**: Route B — switch prediction target from `action[16,7]` to `EEF trajectory[16,3]` to eliminate mapping error

## 3. Key Files

### Core code (third_party/)
- `robomimic/algo/diffusion_policy.py` — diffusion policy + guidance integration + `_guided_scheduler_step()`
- `robomimic/utils/obstacle_guidance_utils.py` — cost functions, `action_chunk_to_eef_xyz_traj()`, pointcloud backprojection
- `robomimic/scripts/run_obstacle_guided_agent.py` — guided rollout entry point
- `robomimic/scripts/run_trained_agent.py` — baseline rollout entry point
- `robosuite/environments/manipulation/pick_place.py` — environment definitions (PickPlaceBreadCan and other distractor variants)

### Diagnostic scripts (scripts/)
- `diagnose_collisions.py` — collision failure diagnosis (EEF-obstacle distance tracking)
- `diagnose_guidance_gradient.py` — denoising step diagnosis (cost/grad_norm/oracle cost logging)
- `calibrate_action_scale.py` — action↔EEF delta calibration
- `benchmark_pointcloud.py` — pointcloud computation overhead benchmark

### Experiment outputs (outputs/)
- `robomimic/eval/baseline/` — baseline (no guidance) rollout results
- `robomimic/eval/obstacle_guided/` — guided rollout results

## 4. File Map

```
AGENTS.md                  ← **You are reading this file**. Project global state + file index
docs/RESEARCH_LOG.md        ← Complete record of discussions, findings, decisions (reverse-chronological append)
.opencode/agents/           ← Agent definitions
  paper-reader.md           ← Paper analysis subagent (Kimi K2.6)
  code-explorer.md          ← Code exploration subagent (DeepSeek V4 Pro)
papers/<name>/              ← Paper PDFs + agent-generated analysis.md
scripts/                    ← Diagnostic + calibration tools
outputs/                    ← Experiment outputs
third_party/robomimic/      ← robomimic fork (exp/pc-obstacle-guidance branch)
third_party/robosuite/      ← robosuite fork (multi-obj-env branch)
```

## 5. Reading Order (for new agents entering)

1. Read this file (AGENTS.md) first
2. For discussion background → `docs/RESEARCH_LOG.md`
3. For paper comparisons → `papers/<name>/analysis.md`
4. For code details → invoke `code-explorer` agent

## 6. Environments

4 PickPlace variants with increasing distractors:
- PickPlaceCan (0) → PickPlaceBreadCan (1) → BreadCerealCan (2) → BreadCerealMilkCan (3)

## 7. Runtime Conventions

- All scripts: `uv run python scripts/...`
- Temporary output: `/tmp/`
- Data path: `robomimic/runs/trained_models/...`
- Python 3.10, managed by uv

## 8. Terminology

- **x0_hat**: estimated clean action/trajectory from current noise during diffusion denoising
- **guidance_geometry_source**: `"pointcloud"` | `"oracle_center"`
- **delta_pos_scale**: linear mapping coefficient from action to EEF delta (from controller config)
- **OSC**: Operational Space Controller (PD controller)
- **cost guidance**: injecting obstacle penetration cost gradients during diffusion denoising to steer trajectories toward safety

## 9. Environment Setup

This repository is a uv-managed Python project. `robomimic` and `robosuite` are
editable source dependencies under `third_party/`.

- Use uv, not conda.
- Python 3.10, pinned by `.python-version`.
- Virtual environment in `.venv/`.
- Do not run scripts with system `python`.

### Sync environment

```bash
uv sync --managed-python
```

### Verify editable imports

```bash
uv run python - <<'PY'
from pathlib import Path
import robomimic
import robosuite

root = Path.cwd().resolve()
print("robomimic:", robomimic.__file__)
print("robosuite:", robosuite.__file__)
assert root / "third_party" / "robomimic" in Path(robomimic.__file__).resolve().parents
assert root / "third_party" / "robosuite" in Path(robosuite.__file__).resolve().parents
PY
```

### Verify PyTorch / CUDA

```bash
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

### Smoke test robosuite (headless)

```bash
MUJOCO_GL=egl uv run python - <<'PY'
import numpy as np
import robosuite as suite
env = suite.make(
    env_name="Lift", robots="Panda",
    has_renderer=False, has_offscreen_renderer=True,
    use_camera_obs=True, camera_names="agentview",
    camera_heights=128, camera_widths=128,
    ignore_done=True, control_freq=20,
)
obs = env.reset()
for _ in range(3):
    action = np.random.randn(*env.action_spec[0].shape)
    obs, reward, done, info = env.step(action)
print("smoke test ok")
env.close()
PY
```

### Run robomimic scripts

```bash
uv run python third_party/robomimic/robomimic/scripts/<script.py> ...
```

### Current source branches

- `third_party/robomimic`: `exp/pc-obstacle-guidance`
- `third_party/robosuite`: `multi-obj-env`

## Maintenance Rules

- **When to update**: when state materially changes (new model trained, root cause confirmed, Route B completed)
- **When NOT to update**: exploratory runs, unconfirmed hypotheses
- **What to update**: overwrite old state directly, delete stale info. Keep file < 1 page
- **Note**: agents may read this file but must not write to it. Maintenance rights belong to humans
