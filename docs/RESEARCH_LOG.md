# Research Log

## Maintenance Rules

- **Format**: reverse chronological, newest on top. Each day is `## YYYY-MM-DD`. Each topic within a day uses `###`.
- **Content**: discussions, findings, decisions with rationale, comparisons with competing works, dead ends.
- **Do NOT write**: raw experiment commands, data directly readable from `outputs/`.
- **When to update**: immediately after each discussion reaches a conclusion. Do not backfill.
- **Marking**: use `[UNVERIFIED]` for hypotheses not yet confirmed.

## Current Unverified Hypotheses

- [UNVERIFIED] After switching prediction target to EEF trajectory, will guidance become effective?
- [UNVERIFIED] Is the Z-axis check (`pen_z`) the root cause of cost ≈ 0?

---

## 2026-06-22

### Core finding: action→trajectory mapping error is the root cause

The cost function uses `cumsum(action * 0.05)` to convert predicted actions to EEF waypoints. Calibration revealed:
- Actual mapping: 3-4cm RMSE vs OSC PD-controller dynamics
- This error is comparable to obstacle radii (3-5cm), making collision cost unreliable
- `delta_pos_scale` = 0.05 (from controller config) is correct for normalized `[-1,1]` actions, but our actions were not normalized (`normalization = null`). Scale=1.0 was 6-300x worse.
- Even the best approximation (0.05) has 3-4cm error — too large for reliable cost computation

### Cost guidance experiment: no improvement across 600 rollouts

- Baseline vs Guided across 4 environments × 3 seeds × 50 rollouts each
- Guidance slightly **reduces** success rates (-3% to -5%) as distractor count increases
- Guidance trigger rate constant at ~12.5% regardless of environment difficulty → not collision-responsive
- PC cost max = 1.13e-04, oracle cost max = 6.66e-04 → both near zero
- Policy's x0_hat does not deeply penetrate obstacles

### Z-axis check in cost function may be a secondary issue

`obstacle_xyz_cylinder_cost` requires both XY penetration AND EEF below obstacle top Z. In pick-and-place tasks, EEF is always above obstacles → `pen_z` always 0 → cost always 0. `obstacle_xy_cost` (XY only) should be tested.

### Decision: Route B — switch prediction target to EEF trajectory

Following from the root cause analysis:
1. Rebuild hdf5 dataset with EEF trajectory labels (not just actions)
2. Modify diffusion policy output from `[7, 16]` to `[3, 16]`
3. Eliminate `action_chunk_to_eef_xyz_traj` conversion entirely
4. Re-train and test whether guidance becomes effective

### Discussion: relation to Lan-o3dp

Lan-o3dp predicts EEF trajectory directly — they never face this mapping error. Our work differs in two ways:
1. We discovered and documented this limitation (they didn't)
2. We provide the Δvis + Δgeo decomposition framework (they don't)

### Agent infrastructure planning

Set up project workspace for multi-agent collaboration:
- `AGENTS.md` as entry point for all agents
- `docs/RESEARCH_LOG.md` for discussion history
- `.opencode/agents/paper-reader.md` (Kimi K2.6) for paper analysis
- `.opencode/agents/code-explorer.md` (DeepSeek V4 Pro) for code exploration

## 2026-06-21 (approximate)

### Earlier discussion: diagnostic tools and failure analysis

Created three diagnostic tools:
- `diagnose_collisions.py`: passive EEF-obstacle distance logging during baseline rollouts
- `diagnose_guidance_gradient.py`: per-denoising-step cost/grad_norm logging
- `calibrate_action_scale.py`: action↔EEF delta mapping verification

### Collision diagnostics: all failures are collision-type

On PickPlaceBreadCerealMilkCan (hardest environment), baseline diagnostics show ALL failures involve collisions (hit_threshold < 3cm). Zero non-collision failures. This confirms the Δvis+Δgeo decomposition: oracle mask solves Part A (visual ambiguity), leaving only Part B (physical obstruction).

### Pointcloud caching optimization

Added static obstacle caching in `run_obstacle_guided_agent.py`. Obstacles are static (placed once at reset), so pointcloud is computed once per episode instead of every step. Saves ~674ms per episode (99.8%). Combined with `guidance_start_step_pct=0.7` (skip early denoising steps) for ~20% end-to-end speedup.

### Training-side approaches discussed

Several directions considered for injecting multi-object information during training:
- Object position vector conditioning (oracle in sim, detector+depth in real)
- Structured depth (target-highlighting in depth channel)
- Training-time geometric augmentation (synthetic obstacles in depth)
- Curriculum learning with increasing distractor density

None chosen yet — blocked on the action→trajectory representation issue.
