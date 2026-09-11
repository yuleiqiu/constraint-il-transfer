# Exact same-observation/noise denoising audit: layout28

2026-09-10. User authorized paired guidance-on/off sampling at the retreat
transition. No new simulator rollout, policy training, or policy modification.

## Exactness

Replay the original seed and saved observation history through each chunk's
sampling call. CPU/CUDA RNG hashes match all33 saved chunk boundaries0--256.
At224/232/240/248/256, the branch matching the original runtime guidance gate
reproduces all seven dimensions of all eight recorded actions with maximum
absolute error **0.0**. Source and forced-on branches use identical restored
RNG state; their sampling consumes the same subsequent CUDA RNG state.
Thus these comparisons use the original noise, not newly selected samples.

## Findings

Mean forward target offset is measured relative to current EEF, projected along
the fixed initial Can-to-goal XY direction. Positive means toward the goal.

| Step | Original guidance active | Source offset cm | Forced-on guidance offset cm |
|---|---|---:|---:|
|224|yes|+1.41|+2.23|
|232|yes|+0.86|+1.22|
|240|no|-2.65|-2.21|
|248|no|-3.65|-3.71|
|256|no|-3.30|-3.22|

At240 onward all eight action targets are backward in both sampled branches.
Forcing the same point guidance on at these fixed states does not restore a
forward command. This is not a closed-loop test of keeping guidance on earlier.

Guidance is active at chunk boundaries120--232 and inactive from240 through
the end. The activation criterion is current Can z > table z +0.08 =0.88m;
Can z at232 is0.94527m and at240 is0.87583m. It is recomputed each chunk,
not latched after initial lift. First actual obstacle contact occurs later,
at256. The earlier interpretation that the *retreating* commands necessarily
had active guidance was incorrect.

### XY cost changes Z output

At232, without guidance, eight EEF Z targets run from0.972 to0.986m (rising).
With guidance they run from0.949 down to0.894m (descending). Maximum same-noise
per-waypoint Z difference is0.092346m. At224 the maximum Z difference is0.036809m.
These are final action targets, not actual achieved displacements.

The implementation differentiates the XY clean-action cost through the denoiser
with respect to the entire noisy action tensor and subtracts the entire update.
It does not constrain final Z, rotation, or gripper channels to match the source.
An XY-only cost therefore does not imply an XY-only final action modification.
The same-input experiment establishes an effect on Z output at these states;
separating the full-gradient update from later denoiser coupling would require
another ablation.

Observed sequence: active guidance changes vertical targets before the turn;
the executed Can descends below the threshold; guidance switches off; the
source at that intervention-produced state outputs backward commands. This
does not prove why the source chooses backward motion or guarantee a successful
rollout if Z is constrained.

### Correction to cloud-anomaly interpretation

The geometric outliers documented at344/376/480/560 remain anomalies in the
saved obstacle cloud, but guidance is **off at all those boundaries**. The
previous nearest-point calculations there were hypothetical selections, not
points applied by active guidance. They cannot explain sustained retreat by
active point repulsion in this rollout. The cloud anomaly needs its own audit.

## Next bounded candidate, not executed

Test a transport guidance variant that preserves task-critical output channels
(especially Z) relative to the same-observation/noise source reference, with
an explicit output check. Merely masking XY gradients does not automatically
guarantee unchanged final non-XY channels after coupled denoising. Separately
test the height-gate switching rule only after distinguishing these effects.
Use task-level rollouts to judge improvements; smoother or forward targets
alone are insufficient. This case motivates an ablation, not a general claim.

## Artifacts

- `results.json`: exact source/forced-on actions, gate, heights and forward offsets.
- `rng_reconstruction.json`:33 original RNG-hash checks.
- `step*_rng.npz`: actual reconstructed RNG states at the five compared steps.
- `step*_diagnostics.json`: forced-on denoising diagnostics.
- `manifest.json`: checkpoint provenance and audit script hash.
- `initial_forced_on_attempt/`: retained failed assumption that guidance was on
  at240; equality assertion stopped that attempt. Final audit checks the actual
  gate and distinguishes forced-on counterfactuals from original execution.
