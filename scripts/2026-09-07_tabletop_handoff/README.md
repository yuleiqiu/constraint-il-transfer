# Tabletop oracle route and handoff diagnostic

Authorized by the user on 2026-09-07 after discussing the five-condition pilot.
This is a diagnostic experiment, not a learned avoidance method. No training,
MVEE resumption, or guidance parameter search is part of this experiment.

Completed and reviewed on 2026-09-08: 45 main branches across layouts 43, 49
and 27, plus three conditional near-goal handoffs on layout 27. All conditional
handoffs remain collision-free but fail the terminal retreat criterion.
See the [final report](../../outputs/tabletop_handoff_diagnostic/conclusion.md).
No further experiment is selected.

## Question and gates

From an already grasped, lifted, contact-free state, does a short prefix of a
safe teacher route suffice for the frozen source to finish, or is helping the
robot past the obstacle necessary under the tested route? Can the source finish
after that longer assistance?

Use the wide-support epoch-50 absolute world EEF policy and saved seed-828
RaisedMilk bank. Screen the seven archived baseline collision failures in
ascending episode order: 8, 17, 25, 37, 43, 48, 49. Preserve every screening
outcome. Eligibility requires a contact-free grasped/lifted chunk boundary and
a fresh baseline collision failure. Select at most three layouts, independently
of the new branch outcomes. This is a selected transport-state diagnostic,
not an estimate of full-task population success.

Screening extension declared after these seven yielded only episodes 43 and 49,
before evaluating the extension's outcomes: inspect the same bank's remaining
three archived contact episodes, 18, 27, 28, in that order. Their old task labels
were successes, but the confirmed scoring issue makes those labels insufficient
for eligibility decisions. The unchanged fresh-baseline gate still applies.
This caps screening at the ten archived contact cases; do not alter obstacles,
source weights or eligibility to obtain three cases. Stop at three eligible
states or exhaustion of these ten candidates and report actual coverage.

Conditions share the actual source-executed prefix and H600 total budget:

1. baseline: frozen source continuation;
2. guided: original point guidance, rho 45, Q 0.08, smoothing 0.01,
   target-lifted activation, XY only, no tangent term;
3. short: eight actions of the teacher route, then source;
4. long: same teacher route until the bypass-past waypoint, then source;
5. teacher: complete route, placement, release and retreat.

First validate the complete teacher. A teacher collision or non-completion
stops evaluation for diagnosis: source-handoff failure is not interpretable
without an executable safe reference. The handoff diagnostic uses oracle
geometry and task phase. Route construction uses a conservative obstacle radius
plus an actor allowance, not a formal collision guarantee. Full physical contact
and task completion determine the gate. Keep all failed teacher probes.

## Replay and observation contract

Replay saved executed actions from the original XML and initial state instead
of restoring a mid-trajectory simulator state with incomplete controller state.
Seed the environment's Generator and Python/NumPy/Torch RNGs. Compare the
resulting simulator state and complete policy observation history against the
captured boundary. Restore policy-sampling RNGs at the boundary. Do not claim
exact replay from seeds alone; report repeated outcomes and numerical audits.

Source chunks are generated directly through the existing paper-guidance sampler,
so there is no stale action queue at handoff. Every teacher step calls env.step,
which updates the real frame history. Keep the original full-pose absolute EEF
action contract and all robot, finger, target and obstacle collision physics.
The three repetitions reuse the same sampling seed to audit reproducibility;
they are not independent policy-noise trials or independent state samples.

Videos are rendered inside the evaluated first repetition, rather than through
a later replay. Later repetitions also audit any rendering-associated drift.
Trace files record EEF and Can paths, contacts, grasp state, goal distance and
actual handoff time. A/B metrics include the shared contact-free prefix.

## Run

Verify CUDA availability and competing GPU workloads first. Keep each layout's
paired conditions and repeats in one process. The primary repetition run uses
GPU 0; after episode 49's first comparison, the independent extension screening
was started on GPU 1 to overlap the remaining repeats. Each process verifies
its own replay prefix; no cross-GPU equality is assumed.

The following is the historical single-candidate preflight command; episode 8
fails the fresh eligibility gate, so it does not reproduce the completed matrix.
Completed run manifests are retained in the report's `replicated/` and
`extension/` directories.

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 ROBOMIMIC_GPU_ID=0 \
uv run python -u scripts/2026-09-07_tabletop_handoff/diagnose.py \
  --episodes 8 --max-states 1 --repeats 1 \
  --out-dir /tmp/tabletop_handoff_preflight
```

The script can resume completed rows in the same output directory. Captured
prefixes and torch RNG states are stored in local pickle files and must only be
loaded from trusted runs. Final reports must state any preflight revisions and
the selected-state / replay limitations. Do not infer that all local guidance
is impossible from a short-versus-long difference along one teacher route.
