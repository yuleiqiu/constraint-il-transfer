# Project Context

## 1. Goal

Single-to-multi-object diffusion policy transfer. Decompose failure modes into two orthogonal dimensions:

- **Part A (Δvis)**: Visual ambiguity — "which object is the target?"
- **Part B (Δgeo)**: Physical obstruction — "arm trajectory blocked by new objects"

## 2. Current State

- Masked-image policy (π_mask) trained on PickPlaceCan (oracle mask input)
- Large-scale experiment (600 rollouts × 4 environments) completed: **inference-time guided obstacle avoidance does not improve success rate**
- **Root cause confirmed**: `action → trajectory` mapping (`cumsum(action * 0.05)`) has 3-4cm RMSE vs. OSC PD-controller actual dynamics, comparable to obstacle radius (3-5cm), making cost guidance unreliable
- **Route B validated and rejected (2026-06-26)**: Tried 4 EEF-based supervision signals (delta_eef_action, next_eef_pos absolute, IK cumulative). All fail open-loop replay with 37-74 cm end error. Detailed report: `docs/route_b_validation/report.md`. The 3-4cm RMSE is **not** an approximation error but the **physical tracking limit** of OSC's force-control PD law at 20 Hz — every EEF-based replay hits the same wall. All robosuite controllers (OSC, IK, JointPosition) are designed for delta commands, not absolute targets.
- **Next step (open)**: re-evaluate strategy. Three options documented in the report:
  1. Multi-task learning (action head for execution + EEF head for guidance) — smallest delta
  2. Custom absolute-position controller — ~50 lines, requires fixing IK nullspace/step-size
  3. Accept 3-4 cm RMSE and improve guidance differently (longer-horizon cost, residual model)

## 3. Subproject AGENTS

- `third_party/robomimic/AGENTS.md` — robomimic architecture, config system, diffusion policy pipeline, guidance components, rollout scripts
- `third_party/robosuite/` — (no separate AGENTS yet; distractor env variants in `robosuite/environments/manipulation/pick_place.py`)

### Project scripts (scripts/)
- `scripts/AGENTS.md` — index of all 7 local scripts (categories, conventions, eval-matrix grouping)

### Experiment outputs (outputs/)
- `robomimic/eval/baseline/` — baseline (no guidance) rollout results
- `robomimic/eval/obstacle_guided/` — guided rollout results
- `route_b_validation/` — EEF-based prediction-target validation (Plans A/B-1/B-2/C) + per-controller verifications

## 4. File Map

```
AGENTS.md                   ← **You are reading this file**. Project global state + file index
docs/RESEARCH_LOG.md        ← Complete record of discussions, findings, decisions (reverse-chronological append)
.opencode/agents/           ← Agent definitions
  paper-reader.md           ← Paper analysis subagent
  code-explorer.md          ← Code exploration subagent
papers/<name>/              ← Paper PDFs + agent-generated analysis.md
scripts/                    ← Diagnostic + calibration tools
outputs/                    ← Experiment outputs
third_party/robomimic/      ← robomimic fork (→ AGENTS.md)
third_party/robosuite/      ← robosuite fork
```

## 5. Reading Order (for new agents entering)

1. Read this file (AGENTS.md) first
2. For discussion background → `docs/RESEARCH_LOG.md`
3. For paper comparisons → `papers/<name>/analysis.md`
4. For robomimic code details → `third_party/robomimic/AGENTS.md`, then invoke `code-explorer` agent

## 6. Environments

4 PickPlace variants with increasing distractors:
- PickPlaceCan (0) → PickPlaceBreadCan (1) → BreadCerealCan (2) → BreadCerealMilkCan (3)

## 7. Runtime Conventions

- All local scripts: `uv run python scripts/...`
- Robomimic scripts: `uv run python third_party/robomimic/robomimic/scripts/<script.py> ...`
- Temporary output: `/tmp/`
- Data path: `robomimic/runs/trained_models/...`
- Python 3.10, managed by uv

## 8. Terminology

- **Part A (Δvis)**: Visual ambiguity — "which object is the target?"
- **Part B (Δgeo)**: Physical obstruction — "arm trajectory blocked by new objects"
- **OSC**: Operational Space Controller (PD controller)
- **EEF**: End-Effector Frame (robot gripper position)
- **Route B**: Switching prediction target from `action[16,7]` to `EEF trajectory[16,3]` to eliminate action→trajectory mapping error
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

- **When to update**: when state materially changes (new model trained, root cause confirmed, Route B completed)
- **When NOT to update**: exploratory runs, unconfirmed hypotheses
- **What to update**: overwrite old state directly, delete stale info. Keep file < 1 page
- **Note**: agents may read this file but must not write to it. Maintenance rights belong to humans
