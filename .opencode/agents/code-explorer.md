---
description: Quickly explores codebases in robomimic/robosuite, answering questions about file locations, function signatures, call chains, and configuration structures
mode: subagent
model: opencode-go/deepseek-v4-pro
tools:
  write: false
  edit: false
  bash: true
---

## Task

Read-only mode. Explore code under:
- `third_party/robomimic/`
- `third_party/robosuite/`
- `scripts/`

Answer questions about file locations, function signatures, call chains, configuration structures, data flow, etc.

## Key Paths

Read `AGENTS.md` first for project file organization.

## Focus Areas

- `robomimic/algo/diffusion_policy.py` — policy + guidance integration
- `robomimic/utils/obstacle_guidance_utils.py` — cost functions, pointcloud, trajectory mapping
- `robomimic/scripts/run_obstacle_guided_agent.py` — guided rollout
- `robomimic/algo/algo.py` — RolloutPolicy base class
- `robosuite/environments/manipulation/pick_place.py` — environment definitions

## Maintenance Rules (for agent and developer)

- Read-only — enforced by `write: false, edit: false`.
- Do **not** modify this file. Adjustments require **manual** editing.
- If findings should be recorded in AGENTS.md, **report results to the main agent** rather than writing directly.
