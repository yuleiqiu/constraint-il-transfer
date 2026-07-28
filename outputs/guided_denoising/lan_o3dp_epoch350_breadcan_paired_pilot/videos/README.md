# Synchronized Baseline / Guided Replays

Each video places the unguided baseline on the left and guidance on the right.
Frames are sampled every two environment steps and encoded at 10 fps. If one
branch terminates earlier, its final frame is held while the other branch
continues.

The comparison uses stronger matching than merely reusing an integer seed:

1. the layout seed reconstructs one simulator state;
2. the reconstructed state's SHA-256 hash must match the state recorded by the
   formal paired experiment;
3. baseline and guided branches both call `reset_to` with that same state;
4. both branches use the same policy / diffusion seed;
5. replay task-success, collision, and observation-failure outcomes must match
   the formal experiment before the video is accepted.

| video | layout seed | policy seed | initial-state hash | transition |
|---|---:|---:|---|---|
| `seed760_episode2_baseline_vs_guided.mp4` | 760002 | 760000002 | `ed10f3a86038b002df82e58f328234375196e1bd1341f2c20a152a275620a081` | safe success → collision failure |
| `seed760_episode5_baseline_vs_guided.mp4` | 760005 | 760000005 | `eabaf3b6305f209e4cf1207355df9c504297abd17178a83e16e98e17a5d62030` | no-collision failure → success with collision |

Machine-readable replay metrics and hashes are in `metadata.json`.

## Agentview point-cloud projection

The `_camera_pointcloud.mp4` variants follow the Camera View presentation in
LAN-O3DP Figure 6. The normal agentview RGB comparison remains on top. The
lower row keeps the same camera viewpoint, dims the RGB scene to gray, and
projects the exact latest-frame 512-point `task_pointcloud` back into
agentview pixels using the runtime camera calibration.

| video | transition |
|---|---|
| `seed760_episode2_baseline_vs_guided_camera_pointcloud.mp4` | safe success → collision failure |
| `seed760_episode5_baseline_vs_guided_camera_pointcloud.mp4` | no-collision failure → success with collision |

Point color encodes world Z from blue (low) to red (high). `QUERY` marks the
eight-step action-chunk sampling boundary. The policy receives two observation
frames; the projection shows the latest frame to keep the camera view legible.
The two visible point sets are the physical `Can` and the `VisualCan` goal
marker. `Bread` is excluded from the policy observation.

Oracle MuJoCo geom segmentation selects `Can_*` and `VisualCan_*` pixels in the
256×256 agentview render. Their depth pixels are unprojected to world XYZ and
deterministically farthest-point sampled to 512 points. The video shows those
raw coordinates before the policy's per-axis min–max normalization; that
normalization changes scale but not the visible geometry or membership.

The shared replay-state hashes, comparison against the recorded pilot, camera
projection description, and replay metrics are in
`camera_pointcloud_metadata.json`.
