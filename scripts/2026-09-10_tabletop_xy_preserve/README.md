# Preserve non-XY source outputs during point guidance

Completed2026-09-10, reviewed2026-09-11. Source/guided/protected Task10/6/9,
Safe9/6/9 of10; expansion gate failed. Canonical output `conclusion.md` has
paired transitions and the separate negative layout28 pilot.

Authorized2026-09-10. Frozen wide-support epoch50, original point cost and
instantaneous target_lifted gate unchanged. The treatment samples source and
original guidance under the same observation and restored Python/NumPy/CPU/
CUDA RNG, executes guidedXY plus sourceZ/axis-angle/gripper, and restores RNG
progress to one sampling call. Final channel identity is asserted every active
chunk. Reference comes from the current treatment state, not a baseline rollout
at the same time. This is output composition, not projected gradients.

Pilot: layout28, original guidance/protected, each twice. Gate: full repeated
trajectory identity, initial observation pairing, exact reproduction of the old
600-step original guided actions, and zero non-XY reference error. Outcome does
not select development layouts. Mechanistic success does not establish task gain.

Fixed development layouts before results:0,5,10,15,20,25,30,35,40,45.
Three conditions each: baseline, original guidance, protected (30 rollouts).
Every condition runs from original initial state atH600; no new training, rho/Q
sweep, obstacle modification, phase latch or cloud filtering. Pilot excluded.
Development set is a systematic subset of a previously examined bank, not heldout.
Continue to a new-layout validation only if protected has at least2 net safe
successes over original guidance, no lower Task SR than original guidance,
and no lower aggregate Task SR than baseline. This small-set gate selects
whether to investigate further; passing it is not evidence of generality.

New success requires physical-goalXY/height and no contact with any gripper
collision geom over11 successive20Hz samples spanning0.5s. No minimum retreat.
Do not terminate on the old strict success; log it diagnostically. Collisions
are counted from episode start through new confirmation. No continuation is
censored by the old scorer. All videos are recorded inside actual execution.

Each branch stores MP4, stepwise states/actions, complete chunk observations,
source/guided/composed chunks, denoising diagnostics, per-step contacts and hold
status, reset audit and result. result.json written last. Original guided source
reference arrays are NaN where not sampled, not fabricated references.
Large arrays/videos/logs remain local; reports and Markdown video index support
VS Code (no HTML). Output: outputs/tabletop_xy_preserve_2026-09-10/.

Use uv and EGL, inspect GPUs first. Pilot onGPU1; after gate, development pairs
remain within a process, splitGPU0 layouts0,10,20,30,40 andGPU1 layouts5,15,25,35,45.
