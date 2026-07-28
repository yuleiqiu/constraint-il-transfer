# Project Context

## 1. Goal

Study single-to-multi-object transfer for a diffusion imitation policy using an
executable delta-EEF action interface.

The current question is whether a small number of targeted, successful
multi-object corrections can inject missing task-preserving avoidance behavior
into the reliable single-object policy without materially degrading its clean
behavior. The immediate experiment is a full-policy targeted fine-tuning upper
bound, not yet the final proposed method.

Deployment-time guided denoising was the preceding research stage. Its
formulation and negative results remain important evidence, but it is no longer
the active implementation direction.

## 2. Current State

- **Targeted fine-tuning upper bound is active (2026-07-28)**: collect a small
  set of complete successful hybrid correction episodes on fresh multi-object
  layouts, initialize from the epoch-260 delta-EEF checkpoint, and train with
  balanced clean replay plus oversampled correction data. Compare against the
  frozen checkpoint and a clean-only continued-training control. See
  `docs/targeted_finetuning/plan.md`.
- **Waypoint teacher feasibility is positive**: on the 20 held-out strict
  pre-collision states, discrete-yaw waypoint search changed Task success from
  1/20 to 18/20 and distractor collision from 20/20 to 2/20, with 17
  failure-to-success and no success-to-failure transitions. This is a selected
  teacher upper bound, not rollout policy evaluation. See
  `scripts/2026-07-27_semi_manual_waypoint_teacher/README.md`.
- **Frozen-policy inference-only methods reached a negative stage conclusion
  (2026-07-27)**: oracle guidance changes trajectories and local clearance, but
  did not provide reliable Task SR gains. At 20 strict states, none of 640
  policy samples was safe and progress-preserving at a 1 cm expanded boundary;
  support-constrained projection was feasible in 1/20 states versus 14/20
  without the support bound. See
  `docs/guided_denoising/frozen_policy_inference_limitations.md`.
- **The reliable source policy is fixed**: use the clean-image
  `delta_eef_pose_action` Diffusion Policy epoch-260 checkpoint. Independent
  Task SR over 0/1/2/3 distractors is 0.907 / 0.707 / 0.253 / 0.200; CR is
  0.000 / 0.213 / 0.647 / 0.680. See
  `outputs/eef_pose_osc_policy/multienv_eval_report.md`.
- **The action interface is validated**: delta-EEF chunks reconstruct executed
  EEF trajectories with 0.131 cm mean position error and 0.253 deg mean
  orientation error over a successful rollout. This replaces the old
  OSC-action forward-model route. See
  `outputs/eef_pose_osc_policy/README.md`.
- **Evaluation remains task-level**: report Task SR, Safe SR, CR, NCR, and the
  four-way outcome partition. Local clearance or support diagnostics are
  mechanism evidence, not substitutes for rollout success.
- **World-model and adapter methods are deferred, not rejected**: first use
  full-policy targeted fine-tuning to establish a learnability upper bound.
  If it succeeds, choose where the new avoidance knowledge should live; if it
  fails, revisit observation, architecture, temporal credit, or data quality.

## 3. Subproject AGENTS

- `third_party/robomimic/AGENTS.md` — robomimic architecture, config system, diffusion policy pipeline, rollout scripts
- `third_party/robosuite/` — (no separate AGENTS yet; distractor env variants in `robosuite/environments/manipulation/pick_place.py`)

### Project scripts (scripts/)
- `scripts/AGENTS.md` — index of local scripts (EEF-pose OSC policy tooling, intent diagnostics, guided-denoising diagnostics, eval-matrix grouping)

### Experiment outputs (outputs/)
- `eef_pose_osc_policy/` — active source-policy conclusion, reconstruction
  diagnostics, and multi-environment baseline
- `policy_manifold_diagnostic/` — frozen-policy support diagnostic
- `guided_denoising/` — completed inference-only experiment summaries
- `task_constrained_guidance/` and `sample_refine/` — completed negative stages
- `forward_model/` — archived OSC-action forward-model summaries

## 4. File Map

```
constraint-il-transfer/             ← Project root (independent git repo)
├── AGENTS.md                        ← This file. Project global state + file index
├── metadata/                        ← Environment metadata
├── docs/                            ← Research artifacts
│   ├── RESEARCH_LOG.md              ← Reverse-chronological log of discussions + decisions
│   ├── targeted_finetuning/          ← Active upper-bound plan
│   ├── guided_denoising/             ← Completed inference-only formulations and results
│   └── forward_model_guidance_next_steps.md ← Archived Δgeo forward-model / ranking handoff and results
├── papers/<name>/                   ← Paper PDFs + agent-generated analysis.md
├── scripts/                         ← Local Python scripts (→ AGENTS.md for index)
├── outputs/                         ← Experiment outputs
│   ├── eef_pose_osc_policy/          ← Active source-policy conclusions and diagnostics
│   ├── policy_manifold_diagnostic/   ← Frozen-policy support evidence
│   └── guided_denoising/             ← Completed inference-only experiment summaries
├── .opencode/agents/                ← Agent definitions (paper-reader, code-explorer)
├── third_party/                     ← Editable source deps (managed independently, see §10)
│   ├── robomimic/                   ← robomimic fork (→ AGENTS.md)
│   └── robosuite/                   ← robosuite fork
└── pyproject.toml + uv.lock + .python-version   ← Python 3.10 deps
```

## 5. Reading Order (for new agents entering)

1. Read this file (AGENTS.md) first
2. For project state + recent discussions → `docs/RESEARCH_LOG.md`
3. For the active experiment contract → `docs/targeted_finetuning/plan.md`
4. For why the frozen-policy stage stopped →
   `docs/guided_denoising/frozen_policy_inference_limitations.md`
5. For the waypoint-teacher upper bound →
   `scripts/2026-07-27_semi_manual_waypoint_teacher/README.md`
6. For EEF-pose OSC training and evaluation → `docs/eef_pose_osc_policy_training.md`
7. For local scripts and how to run them → `scripts/AGENTS.md`
8. For historical inference-only methods → `docs/guided_denoising/`
9. For paper comparisons → `papers/<name>/analysis.md`
10. For inner robomimic code → `third_party/robomimic/AGENTS.md`

## 6. Environments

4 PickPlace variants with increasing distractors:
- PickPlaceCan (0) → PickPlaceBreadCan (1) → BreadCerealCan (2) → BreadCerealMilkCan (3)

## 7. Runtime Conventions

- All local scripts: `uv run python scripts/...`
- Robomimic scripts: `uv run python third_party/robomimic/robomimic/scripts/<script.py> ...`
- Temporary output: `/tmp/`
- Model checkpoints:
  `outputs/robomimic/train/<experiment>/<run>/models/<checkpoint>.pth`
  (not the robomimic default `runs/trained_models/`)
- Python 3.10, managed by uv

## 8. Terminology

- **OSC**: Operational Space Controller (PD controller)
- **EEF**: End-Effector Frame (robot gripper position)
- **Executable full-pose EEF action**: `[eef_pos_world or delta_eef_pos_world(3), eef_quat_site_xyzw delta/absolute axis_angle(3), gripper(1)]` sent to `OSC_POSE`
- **Delta EEF pose action**: `[next_obs/robot0_eef_pos - obs/robot0_eef_pos, axis_angle(R_next_site @ R_obs_site.T), gripper]`; preferred replacement for the old OSC-action forward-model path
- **Guided denoising**: modifying reverse diffusion at deployment time using the gradient of a deployment cost evaluated on the predicted clean action chunk
- **Deployment cost**: a lower-is-better differentiable criterion `C(F(o, A), e*)` used to reweight or guide the pretrained policy distribution
- **Predicted clean action**: the denoiser's estimate `A_0_hat` reconstructed from the noisy action at the current diffusion timestep
- **Effective policy support**: the high-probability local set of action
  trajectories the current policy can produce while preserving task progress;
  diagnostics use an empirical Mahalanobis ellipsoid as a proxy, not strict
  mathematical support
- **Correction episode**: a complete successful hybrid rollout in which the
  frozen policy runs normally, an oracle waypoint teacher intervenes in the
  targeted obstruction phase, and the frozen policy resumes after grasp/lift
- **Targeted fine-tuning upper bound**: full-policy adaptation from epoch260
  with original clean replay and deliberately oversampled correction data
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

- `third_party/` is `.gitignore`d at the root. The root
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

- **When to update**: when state materially changes (new model trained, root cause confirmed, EEF action-interface / forward-model status changed)
- **When NOT to update**: exploratory runs, unconfirmed hypotheses
- **What to update**: keep the active goal, current evidence, reading order,
  runtime contract, and repository workflow accurate. Replace stale current
  state rather than accumulating chronological history here; detailed history
  belongs in `docs/RESEARCH_LOG.md`.
- **Note**: Maintenance rights belong to humans. Agents may change this file only under explicit permission.
