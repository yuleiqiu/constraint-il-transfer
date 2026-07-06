# Project Context

## 1. Goal

Single-to-multi-object diffusion policy transfer. Decompose failure modes into two orthogonal dimensions:

- **Part A (Δvis)**: Visual ambiguity — "which object is the target?"
- **Part B (Δgeo)**: Physical obstruction — "arm trajectory blocked by new objects"

## 2. Current State

- Masked-image policy (π_mask) trained on PickPlaceCan (oracle mask input)
- Large-scale experiment (600 rollouts × 4 environments) completed: **inference-time guided obstacle avoidance does not improve success rate**
- **Root cause confirmed**: `action → trajectory` mapping (`cumsum(action * 0.05)`) has 3-4cm RMSE vs. OSC PD-controller actual dynamics, comparable to obstacle radius (3-5cm), making cost guidance unreliable
- **Route B reopened (2026-07-06)**: Corrected full-pose OSC absolute replay succeeds: `next_obs/robot0_eef_pos + next_obs/robot0_eef_quat_site -> axis_angle + gripper` with `OSC_POSE input_type=absolute, input_ref_frame=world, kp=500` replays all 200 PickPlaceCan demos with 100% final success, mean position error 0.51 cm, and mean orientation error 0.39 deg. Earlier absolute-OSC failure was a diagnostic/action-interface issue, not proof that EEF pose cannot control the robot. Next step: train DP to predict this executable full-pose EEF action. See `docs/route_b_validation/report.md`.
- **OSC forward model validated as trajectory predictor (2026-07-03)**: Learned `f_hat(state, OSC action chunk) -> future EEF xyz trajectory` clearly beats `cumsum(action * 0.05)` on held-out demos and random-action env rollouts. It is useful for action-chunk evaluation, but plugging it into current gradient guidance did not improve rollout success.
- **Action-chunk ranking tested (2026-07-06)**: Safe chunks exist, and the forward model ranks them better than cumsum in same-state diagnostics, but geometry-only ranking did not improve rollout success. This is no longer the primary next step unless Route B full-pose learning fails again. Handoff: `docs/forward_model_guidance_next_steps.md`.

## 3. Subproject AGENTS

- `third_party/robomimic/AGENTS.md` — robomimic architecture, config system, diffusion policy pipeline, guidance components, rollout scripts
- `third_party/robosuite/` — (no separate AGENTS yet; distractor env variants in `robosuite/environments/manipulation/pick_place.py`)

### Project scripts (scripts/)
- `scripts/AGENTS.md` — index of local scripts (diagnostics, forward model, eval-matrix grouping)

### Experiment outputs (outputs/)
- `robomimic/eval/baseline/` — baseline (no guidance) rollout results
- `robomimic/eval/obstacle_guided/` — guided rollout results
- `forward_model/` — OSC action-chunk → EEF trajectory model + validation summaries
- `route_b_validation/` — EEF replay diagnostics, corrected OSC absolute replay, adapter rejection, Panda Mink follow-up

## 4. File Map

```
constraint-il-transfer/             ← Project root (independent git repo)
├── AGENTS.md                        ← This file. Project global state + file index
├── configs/                         ← Per-experiment training/eval configs
├── metadata/                        ← Environment metadata
├── docs/                            ← Research artifacts
│   ├── RESEARCH_LOG.md              ← Reverse-chronological log of discussions + decisions
│   ├── forward_model_guidance_next_steps.md ← Current Δgeo handoff + next experiment
│   └── route_b_validation/          ← Route B reports + controller / adapter validation + corrected OSC replay
├── papers/<name>/                   ← Paper PDFs + agent-generated analysis.md
├── scripts/                         ← Local Python scripts (→ AGENTS.md for index)
├── outputs/                         ← Experiment outputs
│   ├── robomimic/eval/              ← Baseline + obstacle_guided rollouts (600 rollouts)
│   └── route_b_validation/          ← Per-controller verification summaries
├── .opencode/agents/                ← Agent definitions (paper-reader, code-explorer)
├── third_party/                     ← Editable source deps (managed independently, see §10)
│   ├── robomimic/                   ← robomimic fork (→ AGENTS.md)
│   └── robosuite/                   ← robosuite fork
└── pyproject.toml + uv.lock + .python-version   ← Python 3.10 deps
```

## 5. Reading Order (for new agents entering)

1. Read this file (AGENTS.md) first
2. For project state + recent discussions → `docs/RESEARCH_LOG.md`
3. For current Δgeo handoff + next experiment → `docs/forward_model_guidance_next_steps.md`
4. For local scripts and how to run them → `scripts/AGENTS.md`
5. For the Route B experiment report → `docs/route_b_validation/report.md`
6. For paper comparisons → `papers/<name>/analysis.md`
7. For inner robomimic code → `third_party/robomimic/AGENTS.md`, then invoke `code-explorer` agent

## 6. Environments

4 PickPlace variants with increasing distractors:
- PickPlaceCan (0) → PickPlaceBreadCan (1) → BreadCerealCan (2) → BreadCerealMilkCan (3)

## 7. Runtime Conventions

- All local scripts: `uv run python scripts/...`
- Robomimic scripts: `uv run python third_party/robomimic/robomimic/scripts/<script.py> ...`
- Temporary output: `/tmp/`
- Model checkpoints: `outputs/robomimic/checkpoints/<policy>/best.pth` (not the robomimic default `runs/trained_models/`)
- Python 3.10, managed by uv

## 8. Terminology

- **Part A (Δvis)**: Visual ambiguity — "which object is the target?"
- **Part B (Δgeo)**: Physical obstruction — "arm trajectory blocked by new objects"
- **OSC**: Operational Space Controller (PD controller)
- **EEF**: End-Effector Frame (robot gripper position)
- **Route B**: Switching prediction target from `action[16,7]` to `EEF trajectory[16,3]` to eliminate action→trajectory mapping error
- **Executable full-pose EEF action**: `[eef_pos_world(3), eef_quat_site_xyzw -> axis_angle(3), gripper(1)]` sent to `OSC_POSE` in absolute/world mode
- **Forward model**: Learned surrogate `f_hat(state, OSC action chunk) -> future EEF xyz trajectory`, trained on original OSC-action demos and used for action-chunk evaluation
- **Action-chunk ranking**: Proposed next inference-time strategy — sample multiple diffusion action chunks, score predicted EEF trajectories with obstacle geometry, execute the safest chunk without gradient-updating actions
- For diffusion/guidance terminology → `third_party/robomimic/AGENTS.md`

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

## 10. Git Workflow

This directory contains three **independent** git repos. The root repo
only tracks files outside `third_party/`; the two forks are managed
manually.

| Repo | Path | Current branch | Commit style |
|------|------|----------------|--------------|
| root | `./` | `main` | `<scope>: <description>` |
| robomimic | `third_party/robomimic/` | `exp/guided-dp` | `<scope>: <description>` |
| robosuite | `third_party/robosuite/` | `multi-obj-env` | `feat: ...` (Conventional Commits) |

### Rules

- `third_party/` is `.gitignore`d at the root (`.gitignore:6`). The root
  repo **must not** attempt to `git add` files under it.
- Before any `git add` / `git commit`, run `git status` to confirm the
  working directory is the right repo. Each fork has its own `.git/`
  directory; `cd` into the target path first.
- Never push from inside a sub-fork to its upstream without explicit
  confirmation. Upstream sync is handled manually.
- Never force-push (`-f`) on shared branches.
- When a change in a sub-fork is required for the root project to run,
  commit it in the sub-fork first, then re-run `uv sync` and verify.

### Verification before committing

```bash
# From repo root
git status
git check-ignore -v third_party/robomimic third_party/robosuite

# From third_party/robomimic
(cd third_party/robomimic && git status)

# From third_party/robosuite
(cd third_party/robosuite && git status)
```

### Maintenance

- The branch column reflects the current state. Update it when a fork
  switches branches.
- Commit-style column reflects the observed convention. Update only if
  the convention changes.

## Maintenance Rules

- **When to update**: when state materially changes (new model trained, root cause confirmed, Route B / forward-model status changed)
- **When NOT to update**: exploratory runs, unconfirmed hypotheses
- **What to update**: overwrite old state directly, delete stale info. Keep file < 1 page
- **Note**: Maintenance rights belong to humans. Agents may change this file only under explicit permission.
