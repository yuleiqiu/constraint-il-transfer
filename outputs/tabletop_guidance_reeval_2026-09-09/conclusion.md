# Corrected original-guidance evaluation

Completed 2026-09-09; reviewed 2026-09-10. All 100 formal branches on the
original 50 RaisedMilk layouts completed, with one in-execution MP4 per branch.
Eight separate repeated preflight branches are not pooled into the statistics.

## Results

| Condition | Task SR | Safe SR | CR | NCR | Safe / success+contact / collision failure / NCR |
|---|---:|---:|---:|---:|---|
| Source | 28/50 (56%) | 23/50 (46%) | 9/50 (18%) | 18/50 (36%) | 23 / 5 / 4 / 18 |
| Original target-lifted point guidance | 28/50 (56%) | 27/50 (54%) | 7/50 (14%) | 16/50 (32%) | 27 / 1 / 6 / 16 |

Task transitions: 10 failure-to-success and 10 success-to-failure; net zero.
Safe-success transitions: 11 improvements and 7 regressions; net +4/50,
or +8 percentage points. The paired layout-bootstrap 95% interval is
[-8, +24] percentage points; exact two-sided McNemar p=0.481. Task-difference
bootstrap interval is [-18, +18] points; exact p=1.0. These intervals characterize
resampling of this previously examined bank, not a fresh held-out population.

## Interpretation

The corrected comparison shows a positive *observed* net change in safe
completion, alongside substantial paired regressions. It does not establish
a reliable aggregate improvement and does not prove absence of an effect.
Unchanged aggregate Task SR must not be described as preserving every task:
ten originally successful episodes fail under guidance. Lower total contact
also does not imply fewer collision failures: success-with-contact decreases
from five to one, while collision failures increase from four to six.

The measured object of study is this frozen wide-support point-cloud
absolute-EEF policy and this fixed original guidance setting: rho45, Q0.08,
Huber smoothing0.01, XY only, no tangent, target-lifted activation, H600.
There is no evidence here that all guidance methods fail or that a persistent
teacher/corrective option is necessary. No new method or experiment is selected.

## Evaluation audit and boundaries

Goal/obstacle metadata is synchronized to the restored bodies. Each step checks
the true physical-tray XY, height and retreat success predicate. Saved predicate
inputs were also checked offline. Every formal pair matches its initial saved
state and complete initial observation hashes. Within each GPU process,
preflight layouts0/25 reproduce full states, actions, observation hashes,
chunk inputs/RNG hashes and outcomes across repeats; their formal runs also
match those preflights. This is not a repeated-outcome audit of all 50 layouts
or a claim of cross-GPU determinism.

GPU0 evaluated layouts0--24 and GPU1 layouts25--49, keeping each pair in one
process. All source/guided rows start from the original episode initial state;
there is no selected grasped-state prefix or teacher intervention. Videos were
recorded inside the evaluated execution, not by rerunning policies afterward.

These results replace use of the old wide-support baseline/target-lifted scores
for the current fixed-bank question. Differences from old outputs cannot be
attributed solely to the scoring fix: randomness/restoration controls and
in-execution recording were also standardized. The old clean-policy and
always-on rates were not re-evaluated and remain subject to their scoring audit.

## Artifacts

### Follow-up terminal-predicate audit (2026-09-10)

Offline inspection of all 100 terminal records found that all 18 source NCRs
end with Can inside the physical goal XY/height region but fail the retreat
predicate (EEF distance 17.7--21.4 mm versus the required approximately42.4 mm).
Of the 22 source failures including contact, 21 have this terminal signature.
For guidance, only4/16 NCRs (and4/22 failures) have that same signature.
This is a numerical terminal-condition finding, not proof of release, a scoring
bug, or why the policy fails to retreat. The other12 guided NCRs require their
own phase/trajectory inspection. The unchanged strict Task SR aggregates
different failure behavior in the two conditions. Before designing a new
avoidance method around low Task SR, investigate source finishing behavior and
preserve the task contract. No new rollout or training was launched for this
audit. Per-episode evidence: `terminal_predicate_audit.json`.

- `video_index.md`: individual MP4 links for direct VS Code viewing; no HTML.
- `report.md`: all 50 layout outcomes; `summary.json`: aggregates/uncertainty.
- `results.jsonl`: 100 formal branch records.
- `shard_*/episodes/epXX/{baseline,guided}/`: video, actions, dynamic states,
  chunk observations/clouds, per-step predicate inputs and reset audits.
- `shard_*/preflight_pass.json`: separate repeated-execution gates.
- `artifact_integrity.json`: video frame-count checks and per-file SHA256.

Videos and compressed arrays are retained locally and ignored by Git; the
checkpoint and state banks are referenced by path and SHA256 without duplication.
