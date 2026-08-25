# Tabletop MVEE guidance

This branch-local successor begins with the geometry representation gate. It
does not yet implement MVEE fitting or a guidance cost.

The Tabletop environment adds a fixed `backview` based on the camera selected
in the clean `third_party/vlsa-aegis` reference repository at commit
`57b1aef`:

- the reference position `[-0.4, 1.07, 1.25]` metres is retained;
- the optical axis is aimed at world point `[0.0, 0.0, 0.86]`, the middle of
  this task's start-obstacle-goal corridor at carried-object height;
- the resulting MuJoCo quaternion is
  `[0.1447130, 0.1035081, -0.5724828, -0.8003794]` in `wxyz` order.

The reference SafeLIBERO tabletop scene and this project use robosuite's same
default `TableArena` coordinate frame, so the Backview position transfers
directly. Its reference quaternion is not used: that optical axis is almost
horizontal and gives the GoalTray zero visible pixels in this task. The
retargeted view covers the Can, obstacle, GoalTray, and sampled corridor from
table height through 1.10 m. It is added only by `PickPlaceCanTabletop` and its
evaluation variants, not to the global arena.

Render the fixed RaisedMilk dual-view preflight:

```bash
MUJOCO_GL=egl uv run python \
  scripts/2026-08-25_tabletop_mvee_guidance/render_dual_view_preflight.py
```

The figure overlays oracle Can, goal, and obstacle segmentation plus a
projected transport centerline on Agentview and Backview. It then shows both
unprojected obstacle clouds and the single deterministic FPS sample drawn after
fusion. The source policy's existing `task_pointcloud` input remains
single-view and unchanged; the new multi-view provider is opt-in for
deployment-only geometry.
