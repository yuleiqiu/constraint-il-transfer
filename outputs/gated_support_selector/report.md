# Gated Corrective-Support Selector Result

Status: completed development gate, 2026-07-29.

## Question

The frozen source policy lacks local safe-progress support, while targeted
positive BC and SDP add that support but degrade closed-loop task behavior.
Can the two results be combined by keeping the source policy as the default
and invoking the best-Safe-SR SDP checkpoint only under an oracle obstruction
gate?

## Method

The source is the epoch-260 clean delta-EEF policy. The corrective policy is
the SDP checkpoint at 6,000 updates. At every eight-action chunk before target
contact:

1. sample the source chunk with a fixed noise seed;
2. trigger only if that chunk has positive target progress, has swept
   EEF-circle clearance at most 1 cm, starts at least 1 cm from every
   distractor, and no target or distractor contact has occurred;
3. sample 32 chunks from the SDP policy;
4. retain chunks with at least 1 cm swept EEF-circle clearance and no more
   than 0.5 mm target retreat;
5. execute the eligible chunk with minimum RMS deviation from the source in
   source-normalized seven-dimensional action space, or fall back exactly to
   the source chunk if none is eligible;
6. re-evaluate at the next chunk and permanently suppress correction after
   target contact or distractor contact.

The source action queue and corrective action queue remain unmodified during
sampling. Source and gated conditions use the same initial simulator state,
source noise seeds, deterministic image-policy settings, and full 400-step
horizon.

## Fixed-state preflight

The first strict check required each SDP candidate to preserve the source
chunk's target progress within 0.5 mm. All 20 held-out pre-collision states
triggered, but none of 640 SDP samples met both that requirement and the 1 cm
clearance requirement.

This exposes an important qualification to the preceding support diagnostic:
its reported 10/20 direct safe-progress states compare SDP samples with the
SDP policy's own, usually slower nominal sample. They are not
source-progress-equivalent actions.

A correction chunk need not move as quickly toward the target as the
collision-producing source chunk. The deployment preflight therefore used the
weaker, explicit no-substantial-retreat rule above. Under that rule:

| states | gate triggers | states with eligible SDP sample | eligible samples |
|---:|---:|---:|---:|
| 20 | 20 | 11 | 138 / 640 |

For selected chunks, median predicted clearance was 1.23 cm, median target
progress was +0.28 mm, and median source-normalized RMS action deviation was
0.144.

## Source-preservation smoke test

`PickPlaceCan` contains no distractors, so the gate is structurally inactive.
In the one-episode smoke test and four-pair pilot, source and gated conditions
had identical initial-state hashes, source seeds, per-chunk geometry
diagnostics, horizon, task outcome, and reconstruction errors. The four-pair
D0 outcomes were three safe successes and one collision-free non-completion
in both conditions.

## Paired D2 gate

The fixed development gate uses `PickPlaceBreadCerealCan`, seed 600, episode
indices 0--19, and paired initial states.

| condition | Task SR | Safe SR | CR | NCR | safe success / success+collision / collision failure / NCR |
|---|---:|---:|---:|---:|---:|
| source | 0.25 | 0.15 | 0.80 | 0.05 | 3 / 2 / 14 / 1 |
| gated selector | 0.15 | 0.10 | 0.75 | 0.15 | 2 / 1 / 14 / 3 |

The gate triggered on 13 chunks. Eight had no eligible corrective sample and
fell back exactly to source; five executed an SDP chunk. There were no action
clips or observation failures.

Paired transitions were:

- 14 collision failures remained collision failures;
- one pre-existing NCR remained NCR;
- two safe successes remained safe successes;
- one success with collision remained success with collision;
- one safe success became NCR;
- one success with collision became NCR.

There were zero failure-to-success transitions and two success-to-failure
transitions. Four of the five selected corrective chunks avoided contact
during that immediate chunk, but this local safety did not produce reliable
subsequent task recovery. The fifth selected chunk had predicted clearance
1.06 cm yet contacted a distractor during execution, showing the remaining
gap between the swept EEF-circle proxy and physical gripper contact.

## Decision

Reject this source-default, oracle-gated, 32-sample SDP selector. Do not scale
it to D1/D3 or additional seeds and do not tune its threshold or candidate
count on the same 20 pairs.

The combination resolves neither failure in isolation:

- targeted training creates locally selectable avoidance actions, but those
  actions are generally slower than the source and do not encode a reliable
  multi-chunk recovery route;
- the oracle geometry gate detects some local obstruction, but it also fires
  on trajectories that the source would complete safely and cannot predict
  whether a locally safe correction will preserve later grasp and placement.

The remaining missing object is a temporally extended avoidance/grasp/recovery
behavior, not more evidence that a local action can be displaced. The proposed
simulator-branch upper bound was subsequently run on three held-out states:
all 96 SDP chunks avoided immediate contact, but none contacted Can or
completed the task after full source continuation. A ranking-only world model
therefore has no successful branch to select in this pilot. See
`outputs/oracle_branch_value/pilot_3state/result_report.md`.

The positive control remains explicit long-horizon behavior: the waypoint
teacher solves the same three states without collision using 121--144
controlled steps. The next model should generate or represent that multi-step
route, rather than score the existing one-chunk SDP candidates.

## Canonical artifacts

- strict source-relative preflight:
  `outputs/gated_support_selector/fixed_state_preflight_source_relative/`
- no-substantial-retreat preflight:
  `outputs/gated_support_selector/fixed_state_preflight/`
- four-pair D0/D2 pilot:
  `outputs/gated_support_selector/paired_pilot_seed600_n4/`
- 20-pair D2 gate:
  `outputs/gated_support_selector/paired_gate_seed600_n20/`
- implementation:
  `scripts/2026-07-29_gated_support_selector/`
