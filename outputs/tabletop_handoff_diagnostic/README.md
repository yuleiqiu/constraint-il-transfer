# Tabletop oracle-route / handoff diagnostic

Status: main matrix complete, reviewed 2026-09-08. Start with
**[conclusion.md](conclusion.md)** for all three layouts, metrics, interpretation,
selection and the separate reset-scoring defect. No further training is selected.

All 45 main branches completed: layouts 43, 49 and 27, five conditions, three
identical-seed repetitions each. Within every layout/condition, all recorded
trajectories and first source chunks match exactly across repetitions. Baseline
fails with collision in all three layouts; original guidance safely solves 27,
short help safely solves 43, and long help safely solves 43/49. Long help on 27
reaches placement but fails the required retreat. Full teacher solves all three.
The conditional near-goal handoff probe on 27 is reported in conclusion.md.

## First-layout pilot (one execution per condition)

From episode 43's actual source prefix at step 112, the full teacher, eight-step
teacher prefix followed by source, and longer bypass followed by source all
complete without recorded obstacle contact. Source alone collides and fails;
original point guidance avoids contact but does not complete by H600.

| Condition | Task | Safe | Contact | Absolute completion / timeout step |
|---|---:|---:|---:|---:|
| source | 0 | 0 | 1 | 600 |
| original point guidance | 0 | 0 | 0 | 600 |
| 8-action help then source | 1 | 1 | 0 | 351 |
| longer bypass then source | 1 | 1 | 0 | 433 |
| full teacher | 1 | 1 | 0 | 393 |

All five branches reproduce the captured simulator state and complete
observation history with maximum absolute difference zero. The source branch
also reproduces the screened outcome, collision time (188) and final distance.
The short teacher prefix displaces Can by about 5.25 mm laterally before
handoff at step 120; the source then follows a different route. This is a
single selected-state feasibility observation, not a learned method result or
evidence that eight actions suffice generally. It does not support temporal
persistence being necessary in this state. Identical-seed repetitions audit
reproduction; they do not provide independent policy-noise evidence.

The source sampler RNG is restored at the common intervention boundary. Teacher
steps do not consume diffusion samples, so source sample index is aligned to
source invocations, not to absolute time across different help lengths. A
hold/replanning control or absolute-time noise alignment would be needed before
attributing the short branch's benefit specifically to its 5-mm direction.

- Pilot metrics, paths and report: `pilot/report.md`, `pilot/analysis.json`,
  `pilot/ep43_paths.png`.
- Synchronized videos from the actual evaluated branches:
  `pilot/ep43_five_way.mp4`. Finished branches hold their last frame.
- Scripts / exact contracts: `../../scripts/2026-09-07_tabletop_handoff/README.md`.
- Repetition results: `replicated/results.jsonl` (43/49),
  `extension/results.jsonl` (27, plus separately labelled near-goal probe).
- Three synchronized five-condition videos: `pilot/ep43_five_way.mp4`,
  `replicated/ep49_five_way.mp4`, `extension/ep27_five_way.mp4`.

## Confirmed reset-metadata issue

The independent `audit_reset_metadata.py` probe restores two saved Tabletop
layouts with three explicitly controlled environment reset seeds each. All six
dynamic simulator states match the saved state exactly, but `goal_xy` differs
from the restored tray position by 19.9--46.2 mm. `obstacle_xy` also differs
from the restored obstacle position. XML/state hashes do not capture these
Python-side task attributes.

The Tabletop `_can_in_goal` predicate reads `goal_xy`. At predefined positions
at 0.8 and 1.2 times the tray half-width along both axes (plus its center),
synchronizing the metadata changes 16/54 synthetic goal-predicate labels.
This confirms a scoring defect, not a measured correction to old rollout SR.
The target-position probes are synthetic and are not policy/teacher successes.
See `reset_metadata_audit.json`; preliminary center / 2-cm-offset probes are
retained as `reset_metadata_center_probes.json` (0/30 label changes).

The current diagnostic explicitly synchronizes metadata to the restored static
body positions before every branch. Historical Tabletop evaluation scripts
using reset_to without this synchronization require an audit before their
Task/Safe/NCR labels can be treated as definitive. In particular, the archived
wide-support evaluation and checkpoint screen use this reset path. We have not
recomputed their rates. This issue does not explain every observation/trajectory
replay difference and does not establish a defect in other environment families.

The first four archived collision-failure candidates (8, 17, 25, 37) change
outcome under this diagnostic's controlled reset and synchronized metadata:
8/25/37 succeed with contact, while 17 is a collision-free non-completion.
Episode 43 remains a collision failure. Preserve these screening records;
do not attribute all differences to either GPU randomness or metadata alone.

Screening 48/49 subsequently finds 48 safe-successful and 49 collision-failing.
Episode 49 has a valid lifted boundary at step 224 and a safe full-teacher
completion at step 484. To attempt the originally intended third layout, the
same fixed bank's other archived contact episodes (18, 27, 28) are the next
baseline-only eligibility screens, declared before their results. Screening
is capped at these ten contact candidates; record fewer than three eligible
layouts if the cap is exhausted.

## Artifact and implementation boundaries

MP4s are ignored local artifacts. Pickle captures contain trusted local action
prefixes, observations and sampling RNGs. No policy training or fork-source
modification was performed. Initial pilot code was subsequently augmented with
explicit reset-metadata logging, a physical-prefix mismatch guard, and an actual
mesh/primitive support check of the past-obstacle handoff plane. Repeated runs
use those additional checks; the initial pilot handoff proxy used body centers.
