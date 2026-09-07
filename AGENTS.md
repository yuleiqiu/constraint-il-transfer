# Project instructions

Updated 2026-09-07 with human authorization. This is the entry point for
project state and operating rules; detailed evidence lives in linked reports.

## Goal and present decision

Study single-to-multi-object transfer for diffusion imitation policies while
preserving reliable task behavior under avoidance interventions.

No next experiment is currently selected. The latest completed work is the
Tabletop wider-demonstration-support test using original point guidance.
The user confirmed that the preceding MVEE geometry probe was put aside.
The branch name `exp/tabletop-mvee-guidance` does not identify the method
used by later experiments.

Persistent corrective options remain an evidence-supported candidate direction,
but resuming that direction has not been decided. Do not treat historical
plans, an agent recommendation, or a branch name as a new research instruction.
No supervisor opinion or subsequent research decision is recorded here.

## Latest evidence and boundaries

- **Wide support, completed:** original nearest-point guidance on a newly
  trained Tabletop point-cloud absolute-EEF policy. At H600 on 50 RaisedMilk
  states, baseline / target-lifted guidance Task SR is 0.58 / 0.62, Safe SR
  0.52 / 0.54, CR 0.20 / 0.18, and NCR 0.28 / 0.28. Task transitions are
  9 rescues / 7 regressions; the difference is not significant. Always-on
  guidance drops Task SR to 0.18. Replay instability limits causal claims.
  Wider dataset routes do not prove learned policy route coverage.
  Start at `outputs/tabletop_wide_support/conclusion.md`.
- **MVEE, stopped probe:** dual-view obstacle fusion and actor ellipsoids were
  implemented and tested on a few states. Large margins removed contact but
  stalled; no robust task benefit or formal matrix was established.
  See `scripts/2026-08-25_tabletop_mvee_guidance/README.md`.
- **Earlier Tabletop point guidance:** 4/11 eligible safe rescues, all in one
  layout, with six collision-free non-completions. Mechanically active but
  layout-dependent; see
  `outputs/tabletop_guidance/counterfactual_rescue_5scene_5noise/conclusion.md`.
- **Empty2D positive regime:** strong guidance at H600 solves 20/20 mirrored
  offset Circle cases safely, versus 13/20 centered cases. The original
  paper-scale H320 gates remain negative. See
  `outputs/empty2d_guidance/smooth_h600_circle/report.md`.
- **Correction boundary:** SDP D1/H600 improves Task SR 0.67 to 0.78
  (12 rescues / 1 regression); rejected D2 full-policy and one-chunk recipes
  do not improve closed-loop success. Persistent SDP has safe handoff windows
  in 7/20 strict states versus teacher success in 18/20. A correction-only
  temporally extended generator is a candidate to test before a handoff model.
  See `outputs/targeted_finetuning/sdp_result_report.md` and
  `outputs/persistent_correction_oracle/full_20/conclusion.md`.
- **Reliable manipulation source:** clean-image epoch-260
  `delta_eef_pose_action`; Task SR at D0/D1/D2/D3 is
  0.907 / 0.707 / 0.253 / 0.200, CR 0 / 0.213 / 0.647 / 0.680.
  Tabletop point-cloud absolute-EEF and Empty2D sources are distinct.
  See `outputs/eef_pose_osc_policy/README.md`.

## Evaluation and stopped recipes

Report Task SR, Safe SR, CR, NCR, paired failure-to-success /
success-to-failure transitions, and the four-way task/contact partition.
Local clearance, dataset diversity, support proxies, and latent distances are
mechanism diagnostics, not substitutes for successful rollouts.

Equal saved simulator hashes and policy seeds do not establish deterministic
GPU point-cloud replay. Audit observation/randomness and outcome reproduction
before claiming exact paired effects; sharing a process alone is insufficient.

Avoidance can create intervention-induced state OOD even when obstacle content
is excluded from policy input. Do not claim OOD elimination.
See `papers/project-situation.md` for dated paper synthesis, not current plans.

Do not automatically resume stopped MVEE probes, blind point-cost rho/Q sweeps,
rejected D2 BC/SDP mixing, or one-chunk ranking/world-model recipes.
A new experiment requires a stated question and task-level gate. Do not infer
a ban on all future guidance methods from these bounded negative results.

## Reading and navigation

Read this entry first, then only the material required by the task:

1. Latest experiment: `outputs/tabletop_wide_support/conclusion.md`.
2. Research decisions and chronology: newest entries in `docs/RESEARCH_LOG.md`.
3. Results by stage: `outputs/README.md`; follow canonical reports from there.
4. Candidate corrective-option method:
   `docs/recoverability_corrective_option/README.md`.
5. Scripts: `scripts/AGENTS.md`, then the relevant experiment README.
6. Inner robomimic work: `third_party/robomimic/AGENTS.md`.
7. Setup: `docs/runtime_setup.md`; checkpoint changes:
   `docs/checkpoint_retention.md`.

Historical plans retain their original context. Current status here and dated
closure reports take precedence over old future-tense instructions.
Paper-specific material lives in `papers/<name>/`; detailed action training
instructions are in `docs/eef_pose_osc_policy_training.md`.

## Runtime and interfaces

- Use uv, Python 3.10 pinned by `.python-version`, and local `.venv/`.
  Do not use conda or system Python.
- Local scripts: `uv run python scripts/...`.
- Robomimic scripts:
  `uv run python third_party/robomimic/robomimic/scripts/<script.py> ...`.
- Headless simulation uses `MUJOCO_GL=egl`; verify device availability before
  launching work. Temporary artifacts belong in `/tmp/`.
- Checkpoints:
  `outputs/robomimic/train/<experiment>/<run>/models/<checkpoint>.pth`.
- OSC means Operational Space Controller; EEF means End-Effector Frame.
  Executable full-pose actions contain translation (3), axis-angle (3), and
  gripper (1), executed by `OSC_POSE`.
- Delta-EEF targets are next-minus-current world EEF position and
  `axis_angle(R_next_site @ R_obs_site.T)`, plus gripper.
  Verify each experiment's absolute/delta action contract before reuse.
- D0--D3 PickPlace variants add 0--3 distractors; Tabletop and Empty2D are
  separate controlled mechanism environments. Their contracts live in the
  corresponding script READMEs.
- Guided denoising evaluates a differentiable deployment cost on predicted
  clean action chunks. Effective support diagnostics are empirical proxies;
  the old learned OSC-action forward model is archived.

## Three independent Git repositories

| Repository | Path | Commit convention |
|---|---|---|
| root | `./` | `<scope>: <description>` |
| robomimic | `third_party/robomimic/` | `<scope>: <description>` |
| robosuite | `third_party/robosuite/` | `feat: ...` (Conventional Commits) |

The forks are editable source dependencies managed independently.
Root ignores `third_party/`; never stage fork files from root.
Before every add or commit, run status in the target repository.
Check root ignore rules with
`git check-ignore -v third_party/robomimic third_party/robosuite`.
For a required fork change, commit there first, then run `uv sync` and verify
editable imports using `docs/runtime_setup.md`.
Never push a fork upstream without explicit confirmation or force-push a
shared branch. Preserve unrelated user changes.

## Documentation maintenance

Keep this entry short and replace stale state instead of appending history.
Record detailed decisions in `docs/RESEARCH_LOG.md`, experiment evidence in
its conclusion/report, output navigation in `outputs/README.md`, and script
navigation in `scripts/AGENTS.md`. Update those links when a stage closes.
Distinguish observed evidence, hypotheses, recommendations, and human decisions.
Do not promote exploration or an unconfirmed mechanism into settled fact.
Maintenance rights belong to humans; agents may change this file only with
explicit permission. This restructuring was authorized on 2026-09-07.
