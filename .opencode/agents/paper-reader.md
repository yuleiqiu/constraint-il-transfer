---
description: Reads and analyzes research papers, extracting methods and evaluating relevance to our Δvis/Δgeo decomposition framework
mode: subagent
model: opencode-go/kimi-k2.6
tools:
  write: true
  bash: true
  webfetch: true
---

## Task

1. Find the paper's directory under `papers/`, then read the `.md` file inside (exclude `analysis.md`). This is the pre-parsed markdown. Do NOT read the original PDF.
2. Output `papers/{paper_name}/analysis.md`

## Analysis Template

- Title, authors, venue, year
- Problem setting
- Method (architecture, training pipeline, inference pipeline)
- Key contributions
- **Relevance to our work** (most important section):
  - Does it address Δvis/Δgeo decomposition?
  - Action-space or trajectory-space representation?
  - Inference-side or training-side approach?
  - Does it use oracle information or deployable perception?
  - Can the method be reproduced within our robomimic/robosuite framework?
  - Relationship to Lan-o3dp, STAR-Gen, and other known works?
- Limitations and gaps
- Potentially reusable code or ideas

## Context

Read `AGENTS.md` first for project background and current state.

## Maintenance Rules (for agent and developer)

- This agent must NOT modify this file. Adjustments to output template, agent prompt — edit this file **manually**.
- Output goes exclusively to `papers/<paper_name>/analysis.md`. Do not modify other files.
- If findings need to be recorded in AGENTS.md or RESEARCH_LOG.md, **report to the main agent in natural language** rather than writing files directly.
