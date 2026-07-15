# Project Context

## 1. Goal

Single-to-multi-object diffusion policy transfer through deployment-time guided
denoising. Use a pretrained diffusion imitation policy as the action prior and
inject a deployment cost during denoising, without retraining the policy.

The immediate research question is whether guidance over executable delta EEF
pose action chunks can improve multi-object task success while preserving the
task behavior learned from demonstrations.

## 2. Current State

- **Guided denoising adopted as the active direction (2026-07-15)**: the project-level formulation defines a deployment-cost-reweighted diffusion policy and guidance through the predicted clean action. See `docs/guided_denoising/formulation.md`.
- **The active branch has no guidance implementation yet**: the next experiment is to implement denoising-time cost guidance against `delta_eef_pose_action` and evaluate whether it improves Task SR over the unguided baseline without merely shifting collision failures to collision-free non-completion.
- **Delta EEF pose replacement validated (2026-07-08)**: clean-image DP trained on `delta_eef_pose_action` reaches 0.98 PickPlaceCan rollout success. A policy action chunk can be reconstructed into the executed EEF pose trajectory with mean position error 0.131 cm and mean orientation error 0.253 deg over a successful rollout. This is the direct replacement for the old OSC-action forward-model path. See `outputs/eef_pose_osc_policy/README.md`.
- **Delta EEF multi-environment eval completed (2026-07-14)**: one best checkpoint evaluated over 4 environments x 3 eval seeds x 50 episodes reaches Task SR 0.907 / 0.707 / 0.253 / 0.200 as distractors increase, with CR 0.000 / 0.213 / 0.647 / 0.680. Hard-environment failures are dominated by pre-target obstruction, but successful collisions and collision-free placement failures both occur. Future comparisons should report Task SR plus Safe SR / CR / NCR and retain the four-way outcome partition. See `outputs/eef_pose_osc_policy/multienv_eval_report.md`.
- **Absolute EEF pose remains a comparison baseline**: best clean-image abs EEF policy reached 0.82 PickPlaceCan rollout success; delta EEF is the preferred action interface.
- **Old OSC-action guidance did not work end to end**: 600-rollout obstacle-guidance evaluation slightly reduced success, and replacing the inaccurate `cumsum(action * 0.05)` trajectory proxy with a learned forward model did not recover Task SR. The old guidance, forward-model, and action-ranking code is archived and must not be treated as the current delta-EEF guidance experiment.

## 3. Subproject AGENTS

- `third_party/robomimic/AGENTS.md` — robomimic architecture, config system, diffusion policy pipeline, rollout scripts
- `third_party/robosuite/` — (no separate AGENTS yet; distractor env variants in `robosuite/environments/manipulation/pick_place.py`)

### Project scripts (scripts/)
- `scripts/AGENTS.md` — index of local scripts (EEF-pose OSC policy tooling, diagnostics, eval-matrix grouping)

### Experiment outputs (outputs/)
- `robomimic/eval/baseline/` — baseline (no guidance) rollout results
- `robomimic/eval/obstacle_guided/` — guided rollout results
- `forward_model/` — archived OSC-action forward-model result summaries only
- `eef_pose_osc_policy/` — delta EEF pose policy conclusion and trajectory reconstruction diagnostics
- `route_b_validation/` — EEF replay diagnostics, corrected OSC absolute replay, adapter rejection, Panda Mink follow-up

## 4. File Map

```
constraint-il-transfer/             ← Project root (independent git repo)
├── AGENTS.md                        ← This file. Project global state + file index
├── metadata/                        ← Environment metadata
├── docs/                            ← Research artifacts
│   ├── RESEARCH_LOG.md              ← Reverse-chronological log of discussions + decisions
│   ├── guided_denoising/             ← Active formulation and implementation contract
│   ├── forward_model_guidance_next_steps.md ← Archived Δgeo forward-model / ranking handoff and results
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
3. For the active mathematical direction → `docs/guided_denoising/formulation.md`
4. For the implementation contract → `docs/guided_denoising/implementation_plan.md`
5. For current EEF-pose OSC training and evaluation → `docs/eef_pose_osc_policy_training.md`
6. For local scripts and how to run them → `scripts/AGENTS.md`
7. For the Route B experiment report → `docs/route_b_validation/report.md`
8. For archived forward-model / ranking results → `docs/forward_model_guidance_next_steps.md`
9. For paper comparisons → `papers/<name>/analysis.md`
10. For inner robomimic code → `third_party/robomimic/AGENTS.md`

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

- **OSC**: Operational Space Controller (PD controller)
- **EEF**: End-Effector Frame (robot gripper position)
- **Route B**: Switching prediction target from original OSC actions to executable EEF pose actions so policy chunks map directly to EEF trajectories
- **Executable full-pose EEF action**: `[eef_pos_world or delta_eef_pos_world(3), eef_quat_site_xyzw delta/absolute axis_angle(3), gripper(1)]` sent to `OSC_POSE`
- **Delta EEF pose action**: `[next_obs/robot0_eef_pos - obs/robot0_eef_pos, axis_angle(R_next_site @ R_obs_site.T), gripper]`; preferred replacement for the old OSC-action forward-model path
- **Guided denoising**: modifying reverse diffusion at deployment time using the gradient of a deployment cost evaluated on the predicted clean action chunk
- **Deployment cost**: a lower-is-better differentiable criterion `C(F(o, A), e*)` used to reweight or guide the pretrained policy distribution
- **Predicted clean action**: the denoiser's estimate `A_0_hat` reconstructed from the noisy action at the current diffusion timestep
- **Forward model**: Archived learned surrogate `f_hat(state, OSC action chunk) -> future EEF xyz trajectory`; result documents are retained, implementation code is removed

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

| Repo | Path | Commit style |
|------|------|--------------|
| root | `./` | `<scope>: <description>` |
| robomimic | `third_party/robomimic/` | `<scope>: <description>` |
| robosuite | `third_party/robosuite/` | `feat: ...` (Conventional Commits) |

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

- Commit-style column reflects the observed convention. Update only if
  the convention changes.

## Maintenance Rules

- **When to update**: when state materially changes (new model trained, root cause confirmed, Route B / forward-model status changed)
- **When NOT to update**: exploratory runs, unconfirmed hypotheses
- **What to update**: overwrite old state directly, delete stale info. Keep file < 1 page
- **Note**: Maintenance rights belong to humans. Agents may change this file only under explicit permission.
