# Released placement with 0.5-second residence

2026-09-10. User authorized trying placement + release + 0.5-second residence,
without the old minimum EEF retreat distance. This is a new task definition,
applied identically to both conditions; old strict scores remain unchanged.

## Operational definition

At the recorded 20Hz control boundaries, Can must satisfy the original physical
goal XY and height bounds and have **no contact with any collision-enabled
gripper geometry**, continuously for at least0.5 seconds (11 consecutive
boundary samples spanning ten intervals). Contact absence is a conservative
release criterion: unilateral incidental contact also prevents confirmation,
even if the environment's bilateral-fingerpad grasp predicate is false. This
does not require a gripper-open command or an EEF retreat distance. Continuous
means continuous sampled eligibility, not a claim to observe every physics
substep between saved boundaries. No support-contact or velocity requirement
was added, and XY membership retains the original object-center convention.

Success time is the first confirmation boundary; safety counts obstacle contact
only through that time, using the saved cumulative contact records. A rollout
with no confirmation by H600 fails. A rollout ending earlier under the old
success predicate without new confirmation is **censored**, not failed or
successful. No missing continuation is invented.

## Result

| Condition | Confirmed task success | Confirmed failure at H600 | Censored | Task SR bounds | Confirmed safe success | Safe SR bounds |
|---|---:|---:|---:|---:|---:|---:|
| Source | 35 | 4 | 11 | 70%--92% | 29 | 58%--78% |
| Original guidance | 17 | 19 | 14 | 34%--62% | 17 | 34%--60% |

These are deterministic missing-outcome bounds, **not confidence intervals**.
Even if every censored guided rollout succeeded and every censored source
rollout failed, Task SR would be62% versus70%. Thus for this fixed bank and
this conservative operational definition, guidance cannot improve Task SR;
it is at least8 percentage points lower. Exact scores and statistical inference
are not supplied without the missing continuations. Safe SR intervals overlap,
so no similarly definitive safe-rate ordering follows from these bounds.

Confirmed four-way counts (safe / success+contact / collision failure / NCR):
source29 / 6 / 2 / 2, plus11 censored; guidance17 / 0 / 6 / 13, plus14 censored.
Observed contact until new success or record end is9 source and7 guided rows;
for censored no-contact rows future contacts are unknown. These are not complete
CR/NCR estimates. No selected complete-pair significance test is substituted for
the missing full paired outcomes.

## Reconstruction and validation

No policy was loaded and no physics integration or new rollout was executed.
Saved initial XML and per-control-step `(time,qpos,qvel)` states reconstruct
gripper contacts through MuJoCo forward evaluation on CPU.

The original Euler simulator stores integrated qpos while its recorded derived
positions/contacts refer to the preceding final2ms physics substep. A direct
forward of saved qpos therefore disagrees with the recorded derived fields.
The audit reverses that single kinematic integration using saved current qvel,
then performs forward evaluation. It asserts Euler integration and checks
reconstructed Can positions against **every** recorded sample (tolerance1e-8m)
and EEF positions (tolerance1e-7m because recorded EEF uses float32). It does not
rewind or extrapolate a physical trajectory. All100 rows pass these checks and
all100 contain reconstructed bilateral fingerpad grasp at some point, providing
a positive check that grasp contacts are not universally missing.

The initial state at step0 is not rewound. Contact masks come from the archived
XML; the formal runner left hand and finger masks enabled/unchanged. The exact
contact geometry names and duration are saved per step.

## Artifacts and remaining scope

- `results.json`, `summary.json`: new labels and bounds, separate from strict SR.
- `epXX_CONDITION_contacts.json`: per-step contacts and confirmation windows.
- `video_index.md`: new labels linked to the original videos for VS Code.
- `continuation_needed.json`: the25 early-terminated branches needing subsequent
  policy execution, within H600, for exact new-definition rates.

No rerun was started. Additional execution would need to reproduce each saved
prefix and then continue the unchanged policy under the new termination rule;
simply appending repeated final frames would not validate residence or safety.
The task-level negative ordering above does not depend on resolving those25
branches, and is not a claim about all possible guidance methods.
