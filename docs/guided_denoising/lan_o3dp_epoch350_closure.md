# LAN-O3DP 40-Demo Epoch-350 Closure

Status: completed negative closure result, 2026-07-27.

## 1. Decision

The LAN-aligned 40-demo, seed-42 point-cloud policy is not an acceptable source
policy for the project, and the tested LAN-style point-level guidance does not
pass the paired method gate.

Do not:

- use the epoch-350 checkpoint as the final source policy;
- tune another guidance scale, safety margin, or late-denoising cutoff for the
  same point-level EEF penetration cost;
- scale this configuration to the full environment matrix or multiple policy
  seeds.

For the next LPB study, freeze the earlier clean-image delta-EEF Diffusion
Policy epoch-260 checkpoint as the source prior. Its independently evaluated
single-object Task SR is 0.907, and its executable action chunks already have
validated EEF trajectory reconstruction.

This result rejects the tested checkpoint and guidance configuration. It does
not establish that every LAN-O3DP component or every deployment-time
conditioning method is ineffective.

## 2. Checkpoint and Questions

Checkpoint:

```text
outputs/robomimic/train/lan_o3dp_can_delta_pose_eps_residual_40demo_seed42/
20260724165723/models/
model_epoch_350_image_v15_delta_eef_pose_osc_pointcloud_lan40_seed42_success_0.65.pth
```

The checkpoint uses:

- 40 selected clean demonstrations;
- target-object point-cloud observation plus proprioception;
- executable 7-D `delta_eef_pose_action`;
- DDPM epsilon prediction with 100 train and inference steps;
- 2 / 16 / 8 observation, prediction, and action horizons.

The closure experiment answered two questions:

1. Does the training-time 20-rollout value of 0.65 reproduce under a fixed,
   independent 150-episode clean evaluation?
2. After adapting the existing predicted-clean-action guidance to this
   point-cloud DDPM policy, does one calibrated configuration improve paired
   task outcomes in both hard clutter levels?

## 3. Independent Clean Evaluation

The frozen checkpoint was evaluated on `PickPlaceCan` with seeds 600, 601, and
602, using 50 episodes per seed and horizon 400.

| eval seed | successes | episodes | Task SR |
|---:|---:|---:|---:|
| 600 | 27 | 50 | 0.540 |
| 601 | 27 | 50 | 0.540 |
| 602 | 21 | 50 | 0.420 |
| **pooled** | **75** | **150** | **0.500** |

The pooled Wilson 95% interval is `[0.421, 0.579]`. The three seed-level rates
have mean 0.500 and sample standard deviation 0.069.

Decision: the checkpoint fails the project's approximately 0.80
single-object source-competence gate. The training-time value of 0.65 is not a
reliable source-policy estimate.

## 4. DDPM Guidance Adaptation and Calibration

The guided checkpoint loader now maps a LAN-O3DP checkpoint to a registered
guided LAN variant while preserving LAN's linear observation-feature fusion.
The reverse-step implementation accepts scheduler outputs containing a
predicted clean sample and follows the actual scheduler timestep sequence, so
the same tested utility supports the existing DDIM policy and this DDPM-100
policy. Disabled guidance reproduces the LAN policy output exactly under the
same observation and random seed.

The calibrated configuration used:

- LAN-equivalent unsquared XY penetration cost;
- oracle distractor centers and radii plus a 2 cm safety margin;
- guidance only at DDPM training timesteps `t <= 19`;
- a 3 cm per-waypoint reverse-step displacement cap;
- the same configuration in the two- and three-distractor environments.

Ten independent fixed states were captured where the unguided predicted
trajectory activated the deployment cost. Common diffusion noise isolated the
effect of scale.

| scale | predicted raw clearance | executed raw clearance | maximum one-step waypoint displacement |
|---:|---:|---:|---:|
| 0 | 1.45 cm | 1.72 cm | 0 |
| 0.0003 | 1.52 cm | 1.78 cm | 0.031 cm |
| 0.001 | 1.70 cm | 1.92 cm | 0.105 cm |
| 0.003 | 2.01 cm | 2.15 cm | 0.315 cm |

Scale 0.003 was frozen for the paired pilot because it was the largest tested
setting, made the mean 2 cm safety-margin clearance positive, and retained
small physical updates. The maximum action-limit excess was `1.50e-05` in
both the scale-zero and guided paths, so the large raw clip counts in the
diagnostic are numerical boundary effects rather than meaningful saturation.

During state capture, some multi-object episodes produced no visible target
pixels and therefore an empty target point cloud. The diagnostic and paired
scripts now record this as an observation failure instead of crashing.
Observation failures did not occur in the final 20 paired layouts, so they do
not explain the paired result; they remain a separate robustness limitation of
the target-only point-cloud representation.

## 5. Paired Closed-Loop Pilot

The pilot used ten matched layouts in each hard environment. Baseline and
guided branches shared the initial simulator state and policy random seed.
Only guidance differed.

| environment | condition | Task SR | Safe SR | CR | NCR |
|---|---|---:|---:|---:|---:|
| 2 distractors | baseline | 0.400 | 0.300 | 0.400 | 0.300 |
| 2 distractors | guided | 0.400 | 0.300 | 0.500 | 0.200 |
| 3 distractors | baseline | 0.800 | 0.600 | 0.400 | 0.000 |
| 3 distractors | guided | 0.600 | 0.400 | 0.400 | 0.200 |

In the two-distractor environment, guidance changed no task outcome and
converted one collision-free failure into a collision failure. In the
three-distractor environment, it converted two baseline successes into
collision-free non-completions. Across both environments:

- baseline failure to guided success: 0;
- baseline success to guided failure: 2;
- Task SR improved in neither environment;
- the fixed gate was not met.

The high three-distractor baseline rate is a property of these ten small-pilot
layouts and must not replace the existing 150-episode baseline estimate. The
paired deltas, not the pilot's marginal baseline rate, decide this elimination
test.

### PickPlaceBreadCan follow-up

A subsequent one-distractor check used 20 matched `PickPlaceBreadCan` layouts
with the same frozen checkpoint and calibrated guidance configuration.

| condition | Task SR | Safe SR | CR | NCR |
|---|---:|---:|---:|---:|
| baseline | 0.450 | 0.400 | 0.050 | 0.550 |
| guided | 0.450 | 0.350 | 0.150 | 0.500 |

One baseline collision-free failure became a guided success with collision,
while one baseline safe success became a guided collision failure. Thus the
task-success gains and losses cancel, Safe SR falls by 0.05, and CR rises by
0.10. There were no observation failures. This simpler environment therefore
does not rescue the frozen point-cloud guidance configuration.

## 6. Interpretation and Handoff

The mechanism is active: it changes DDPM samples, improves fixed-state
predicted clearance, and improves the executed short-prefix clearance. The
negative result is therefore not caused by a disabled update or a broken
delta-EEF reconstruction path.

The failure occurs at the task level. Local point clearance does not represent
the mixed failure mechanisms already identified by the intent diagnostic, and
the guided trajectory can disturb grasp, approach, or placement behavior
without creating a successful alternative. This is consistent with the prior
RGB-policy mechanism baseline and sample-refine experiments: geometric changes
alone have not produced a repeatable task-level gain.

The next activity is an LPB-style method study using the reliable clean-image
delta-EEF Diffusion Policy as the frozen source prior. Before implementation,
the LPB paper should be mapped onto this project's executable action chunk,
observation interface, deployment information, and four-way evaluation gate.
The resulting method should not be described as a LAN-O3DP reproduction.

## 7. Canonical Artifacts

- Clean evaluation:
  `outputs/robomimic/eval/lan_o3dp_epoch350_clean/report.md`
- Fixed-state calibration:
  `outputs/guided_denoising/lan_o3dp_epoch350_same_state/summary.json`
- Paired pilot:
  `outputs/guided_denoising/lan_o3dp_epoch350_paired_pilot/report.md`
- PickPlaceBreadCan paired follow-up:
  `outputs/guided_denoising/lan_o3dp_epoch350_breadcan_paired_pilot/report.md`
- Reliable source-policy baseline:
  `outputs/eef_pose_osc_policy/multienv_eval_report.md`
