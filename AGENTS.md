# Project Context

## 1. Goal

Study single-to-multi-object transfer for a diffusion imitation policy using an
executable delta-EEF action interface.

The current question is how to select and localize targeted avoidance
adaptation without disrupting the task modes of the reliable single-object
policy. Positive-only and set-supervised full-policy fine-tuning both inject
local safe-progress support but fail closed loop.

The active method hypothesis is a temporally persistent corrective option that
returns control when the frozen source policy becomes safely recoverable.
Deployment-time guided denoising was the preceding research stage; its
formulation and negative results remain important evidence.

The active mechanism-isolation experiment is `empty2d_guidance`: it asks
whether the paper-aligned predicted-clean point cost can work at all when EEF,
robot, and obstacle geometry are reduced to an idealized 2-D point setting.

## 2. Current State

- **The manipulation boundary is a persistent corrective option, not local
  chunk selection**: persistent SDP creates non-monotonic source-recoverability
  windows in 7/20 strict D2 states, including five that require multiple
  correction chunks. The other 13 states have zero recoverable trajectories,
  while the waypoint teacher solves 18/20. Learn a correction-only,
  temporally extended object-relative route before a handoff estimator. Do not
  train a general world/value model merely to rank the current one-chunk
  candidates. See `outputs/persistent_correction_oracle/full_20/conclusion.md`.
- **Correction training has a bounded positive regime and a D2 capacity
  boundary**: fixed SDP-6k at D1/H600 changes Task SR from `0.67` to `0.78`,
  with 12 failure-to-success and one success-to-failure transition (exact
  paired `p=0.0034`). In D2, positive BC, lower-exposure BC, SDP, and an
  oracle-gated selector all add local avoidance support but fail to improve
  closed-loop Task SR. Preserve D1/H600 as the positive result; do not scale
  the rejected D2 recipes. See
  `outputs/targeted_finetuning/d1_horizon_salvage_report.md` and
  `outputs/targeted_finetuning/sdp_result_report.md`.
- **Empty2D data and implementation gates pass; formal training is pending
  (2026-08-07)**: the old Toy package and artifacts were deleted in favor of a
  minimal 2-D absolute-target policy with hidden fixed goal and oracle-only
  obstacle boundaries. The fixed 100-trajectory conditional-RBF dataset passes
  all gates: Circle has 33 safe / 57 collision training paths with 19/14
  upper/lower routes; Square has 20 / 70 with 11/9 routes. HDF5 verification,
  24 mechanism/backward-compatibility tests, and a real batch-256 CUDA update,
  reload, and online-action smoke pass. No 500-epoch source or paired guidance
  result exists yet. See `outputs/empty2d_guidance/README.md`.
- **Paper-aligned D2 point guidance is archived as a negative result**: on ten
  fixed D2 states, neither `Q=0.04` nor expanded `Q<=0.12` clears any of four
  gripper collisions. All contacts are gripper/Cereal rather than arm or
  carried-target contact. This motivates, but does not prove, the Empty2D
  geometry-mismatch test. See
  `outputs/lan_o3dp_paper_aligned/d2_guidance_report.md`.
- **OOD success is not OOD elimination**: LAN-O3DP / DP3 primarily reduce
  exogenous environment or observation shift. Avoidance guidance can itself
  create endogenous intervention-induced state OOD by moving the robot away
  from demonstrated trajectories. No reviewed paper eliminates OOD; LPB
  steers toward an expert manifold and ReGuide can absorb specific successful
  recovery states through retraining. Evaluate whether control reaches a
  frozen-source-recoverable state, not whether obstacle content was removed
  from the policy input. See `papers/project-situation.md`.
- **The reliable manipulation source and action interface remain fixed**: use
  the clean-image epoch-260 `delta_eef_pose_action` policy. Task SR over
  0/1/2/3 distractors is `0.907 / 0.707 / 0.253 / 0.200`; CR is
  `0.000 / 0.213 / 0.647 / 0.680`. Delta-EEF chunks reconstruct executed EEF
  motion with 0.131 cm mean position and 0.253 deg mean orientation error. See
  `outputs/eef_pose_osc_policy/README.md`.
- **Evaluation remains task-level**: report Task SR, Safe SR, CR, NCR, paired
  failure-to-success / success-to-failure transitions, and the four-way
  outcome partition. Local clearance, support, and latent-distance diagnostics
  are mechanism evidence, not substitutes for rollout success.

## 3. Subproject AGENTS

- `third_party/robomimic/AGENTS.md` — robomimic architecture, config system, diffusion policy pipeline, rollout scripts
- `third_party/robosuite/` — (no separate AGENTS yet; distractor env variants in `robosuite/environments/manipulation/pick_place.py`)

### Project scripts (scripts/)
- `scripts/AGENTS.md` — index of local scripts (EEF-pose OSC policy tooling, intent diagnostics, guided-denoising diagnostics, eval-matrix grouping)

### Experiment outputs (outputs/)
- `eef_pose_osc_policy/` — active source-policy conclusion, reconstruction
  diagnostics, and multi-environment baseline
- `policy_manifold_diagnostic/` — frozen-policy support diagnostic
- `targeted_finetuning/` — correction-data gates, checkpoint screen, and
  positive-BC / support conclusion
- `gated_support_selector/` — fixed-state and paired-rollout localization gate
- `oracle_branch_value/` — cloned-simulator candidate-existence pilot
- `persistent_correction_oracle/` — temporal source-recoverability pilot
- `empty2d_guidance/` — active idealized point-geometry mechanism test
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
│   ├── targeted_finetuning/          ← Paired-data contract and positive-BC result
│   ├── guided_denoising/             ← Completed inference-only formulations and results
│   └── forward_model_guidance_next_steps.md ← Archived Δgeo forward-model / ranking handoff and results
├── papers/<name>/                   ← Paper PDFs + agent-generated analysis.md
├── scripts/                         ← Local Python scripts (→ AGENTS.md for index)
├── outputs/                         ← Experiment outputs
│   ├── eef_pose_osc_policy/          ← Active source-policy conclusions and diagnostics
│   ├── policy_manifold_diagnostic/   ← Frozen-policy support evidence
│   ├── targeted_finetuning/          ← Correction gates, screen, and support result
│   ├── gated_support_selector/       ← Gated SDP selector result
│   ├── oracle_branch_value/          ← Oracle continuation-existence result
│   ├── persistent_correction_oracle/ ← Persistent correction handoff result
│   ├── empty2d_guidance/              ← Active point-geometry guidance test
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
3. For the cross-paper OOD and project-gap synthesis →
   `papers/project-situation.md`
4. For the active corrective-option hypothesis →
   `docs/recoverability_corrective_option/README.md`
5. For the persistent-correction oracle result →
   `outputs/persistent_correction_oracle/full_20/conclusion.md`
6. For the current Empty2D mechanism test →
   `outputs/empty2d_guidance/README.md`
7. For the oracle branch-value result →
   `outputs/oracle_branch_value/pilot_3state/result_report.md`
8. For the completed gated selector result →
   `outputs/gated_support_selector/report.md`
9. For the completed SDP and exposure-control result →
   `outputs/targeted_finetuning/sdp_result_report.md`
10. For the positive-BC predecessor →
   `outputs/targeted_finetuning/upper_bound_report.md`
11. For why the frozen-policy stage stopped →
   `docs/guided_denoising/frozen_policy_inference_limitations.md`
12. For the waypoint-teacher upper bound →
   `scripts/2026-07-27_semi_manual_waypoint_teacher/README.md`
13. For EEF-pose OSC training and evaluation → `docs/eef_pose_osc_policy_training.md`
14. For local scripts and how to run them → `scripts/AGENTS.md`
15. For historical inference-only methods → `docs/guided_denoising/`
16. For paper comparisons → `papers/<name>/analysis.md`
17. For inner robomimic code → `third_party/robomimic/AGENTS.md`
18. Before deleting local checkpoints → `docs/checkpoint_retention.md`

## 6. Environments

4 PickPlace variants with increasing distractors:
- PickPlaceCan (0) → PickPlaceBreadCan (1) → BreadCerealCan (2) → BreadCerealMilkCan (3)

Empty2D mechanism environments use workspace `[-1,1]^2`, start
`(-0.875,0)`, goal `(0.875,0)`, and either no obstacle, a radius-0.3 circle,
or a centered 0.6-by-0.6 square.

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
