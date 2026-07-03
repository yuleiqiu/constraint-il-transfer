# Panda Mink IK Controller Replay

Dataset: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5`
Controller: `/home/yulei/codes/constraint-il-transfer/third_party/robosuite/robosuite/controllers/config/default/composite/panda_mink_ik.json`
Split: `valid`
Demos: `demo_11, demo_126, demo_135, demo_140, demo_145, demo_166, demo_175, demo_179, demo_180, demo_194, demo_196, demo_24, demo_26, demo_27, demo_50, demo_55, demo_66, demo_69, demo_91, demo_99`

## Summary

| desired cm | actual cm | tracking | mean target err cm | end orig err cm | success any | success final |
|---:|---:|---:|---:|---:|---:|---:|
| 1.681 | 0.358 | 0.215 | 1.324 | 0.267 | 0.950 | 0.950 |

Controller-validation interpretation:

```text
success_final high and end orig error < 1 cm indicate that expert EEF
targets can replay the task through this controller.

mean target error is a lag diagnostic, not a strict pass / fail metric,
because the controller may lag per-step targets while still ending on the
demonstration trajectory and completing the task.
```
