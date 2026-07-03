# Panda Mink IK Controller Replay

Dataset: `/home/yulei/codes/constraint-il-transfer/third_party/robomimic/datasets/can/yq/image_v15_delta_eef.hdf5`
Controller: `/home/yulei/codes/constraint-il-transfer/third_party/robosuite/robosuite/controllers/config/default/composite/panda_mink_ik.json`
Split: `train`
Demos: `demo_1, demo_10, demo_100, demo_101, demo_102, demo_103, demo_104, demo_105, demo_106, demo_107, demo_108, demo_109, demo_110, demo_111, demo_112, demo_113, demo_114, demo_115, demo_116, demo_117, demo_118, demo_119, demo_12, demo_120, demo_121, demo_122, demo_123, demo_124, demo_125, demo_127, demo_128, demo_129, demo_13, demo_130, demo_131, demo_132, demo_133, demo_134, demo_136, demo_137, demo_138, demo_139, demo_14, demo_141, demo_142, demo_143, demo_144, demo_146, demo_147, demo_148`

## Summary

| desired cm | actual cm | tracking | mean target err cm | end orig err cm | success any | success final |
|---:|---:|---:|---:|---:|---:|---:|
| 1.713 | 0.361 | 0.215 | 1.353 | 0.264 | 0.940 | 0.940 |

Controller-validation interpretation:

```text
success_final high and end orig error < 1 cm indicate that expert EEF
targets can replay the task through this controller.

mean target error is a lag diagnostic, not a strict pass / fail metric,
because the controller may lag per-step targets while still ending on the
demonstration trajectory and completing the task.
```
