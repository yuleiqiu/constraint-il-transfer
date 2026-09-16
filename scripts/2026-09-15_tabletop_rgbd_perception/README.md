# Tabletop RGB-D perception comparison

Status: completed and stopped at the pilot gate. Oracle/perceived Task SR was
10/10 versus 0/10 in both clean and RaisedMilk; the gate failed and no full50
expansion, planner run, or policy training was launched. Start with
`outputs/tabletop_rgbd_perception_2026-09-15/conclusion.md`.

This stage compared the frozen wide-support epoch-50 absolute-EEF policy under
its existing oracle `task_pointcloud` and a GroundingDINO + SAM + Cutie RGB-D
replacement.

## Fixed contract

- Environment banks: clean `PickPlaceCanTabletop`, seed 827; obstacle
  `PickPlaceCanTabletopRaisedMilk`, seed 828.
- Pilot layouts: `0 5 10 15 20 25 30 35 40 45`; H600; old strict success OR
  released placement continuously held for 0.5 seconds.
- Conditions run sequentially within each pair on one GPU:
  `oracle_mask_source`, then `perceived_mask_source`, with identical saved
  state and policy seed.
- Perceived runtime input is limited to agentview RGB, metric depth, fixed
  camera calibration, and locked role strings. The inner robomimic oracle
  point-cloud provider is disabled before every perceived reset.
- MuJoCo segmentation, object poses, contacts, and release state are consumed
  only by the separate evaluation/audit path. They never enter
  `PerceptionProvider` or the policy observation.
- DINO box/text thresholds are fixed at 0.35/0.25. Detection and SAM run once
  at episode initialization. Cutie propagates the same instance IDs thereafter;
  there is no detector retry or mid-episode reinitialization.
- Missing `MilkSupport` is nonfatal and recorded as `support_detected=false`.
  Missing/invalid target or goal is a task-perception failure and stops that
  branch. Missing/invalid Milk makes `obstacle_valid=false` but does not stop
  this comparison because obstacle points are not a frozen-policy input.

The implementation follows the components and default Cutie API used by
Lan-o3dp revision `b89c2898d8548f7dfde63c44eeb733a59f0c5222`. Git dependencies
are pinned in the root `pyproject.toml` / `uv.lock`. Unlike the reference real
robot code, formal execution never auto-downloads models.

## Setup and immutable weights

Install the optional uv dependencies:

```bash
CUDA_VISIBLE_DEVICES='' uv sync --extra perception
```

The empty build-time CUDA visibility keeps GroundingDINO on its Lan-o3dp CPU
path and avoids requiring a local CUDA compiler. It does not affect formal
commands, which expose a GPU normally for SAM, Cutie, and the frozen policy.

Download each model once and write file sizes/SHA256 plus repository revisions.
The manifest also includes the two ResNet initialization files loaded by
Cutie's official default constructor; the adapter redirects those lookups to
the verified local copies:

```bash
uv run --extra perception python \
  scripts/2026-09-15_tabletop_rgbd_perception/weights.py \
  --directory outputs/tabletop_rgbd_perception_2026-09-15/weights --download
```

Every later command verifies that manifest and sets Hugging Face/Transformers
offline mode before model construction.

## Prompt calibration and GPU smoke

Choose one prompt per role on the ten fixed RaisedMilk reset scenes, maximizing
mean oracle-mask recall with mean precision and then listed candidate order as
tie-breakers:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
uv run --extra perception python \
  scripts/2026-09-15_tabletop_rgbd_perception/calibrate_prompts.py \
  --weights outputs/tabletop_rgbd_perception_2026-09-15/weights \
  --out outputs/tabletop_rgbd_perception_2026-09-15/calibration
```

Then test one DINO→SAM initialization, two Cutie propagation frames, identical
input repeatability, a frozen-policy action chunk, and peak combined GPU memory:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
uv run --extra perception python \
  scripts/2026-09-15_tabletop_rgbd_perception/smoke_test.py \
  --weights outputs/tabletop_rgbd_perception_2026-09-15/weights \
  --locked-config outputs/tabletop_rgbd_perception_2026-09-15/calibration/locked_config.json \
  --out outputs/tabletop_rgbd_perception_2026-09-15/smoke
```

## Formal pilot and gate

Each environment command produces 20 rollouts. They may use separate GPUs;
each oracle/perceived pair remains inside one process and GPU.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
uv run --extra perception python \
  scripts/2026-09-15_tabletop_rgbd_perception/run.py \
  --environment clean --stage pilot \
  --output outputs/tabletop_rgbd_perception_2026-09-15 \
  --weights outputs/tabletop_rgbd_perception_2026-09-15/weights \
  --locked-config outputs/tabletop_rgbd_perception_2026-09-15/calibration/locked_config.json

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
uv run --extra perception python \
  scripts/2026-09-15_tabletop_rgbd_perception/run.py \
  --environment raised_milk --stage pilot \
  --output outputs/tabletop_rgbd_perception_2026-09-15 \
  --weights outputs/tabletop_rgbd_perception_2026-09-15/weights \
  --locked-config outputs/tabletop_rgbd_perception_2026-09-15/calibration/locked_config.json

uv run python scripts/2026-09-15_tabletop_rgbd_perception/analyze.py \
  --output outputs/tabletop_rgbd_perception_2026-09-15
```

Expansion is permitted only if `pilot_gate.json` says `passed: true`. Run the
same two commands with `--stage expansion`; they execute exactly the remaining
40 layouts and reuse the pilot rows. Re-run `analyze.py` to create the full50
decision. A failed gate stops this stage: it does not authorize fine-tuning,
planner integration, or additional obstacle experiments.

## Tests and artifacts

```bash
uv run pytest -q scripts/2026-09-15_tabletop_rgbd_perception
```

Every rollout stores an actual execution MP4, initial RGB-D and masks, policy
and RNG hashes, states/actions, stepwise perception/audit JSONL, and perceived
task/target/goal/obstacle clouds at policy chunks. `analyze.py` reports mask
IoU/precision/recall, 5/10 mm 3-D nearest-neighbor precision/recall, Chamfer,
centroid/extents error, support detection, identity swaps, tracking failures,
latency, Task/Safe/CR/NCR, the four-way partition, and paired transitions.
Absolute-EEF contract validation uses the executable representation (finite
7-D world XYZ, canonical axis-angle, and normalized gripper). The generic
robosuite `env.action_spec` is retained only as an exceedance diagnostic: its
normalized bounds are not valid bounds for absolute world pose components.
