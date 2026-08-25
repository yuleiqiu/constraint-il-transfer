# Typical 2x2 rollout videos

Every video uses one serialized formal seed-812 state and shows:

```text
full physics / baseline       full physics / guided
hand collision off / baseline hand collision off / guided
```

Panel annotations report the actual rendered replay. `collision seen` is
cumulative, and `final contacts` separates hand, finger, arm, and carried
target contact.

## 1. D2 episode 4: hand-shell obstruction, then residual finger contact

[`episode_004_2x2.mp4`](d2_replay/episode_004_2x2.mp4)

- Full physics: baseline and guided both collide and fail.
- Hand off: baseline and guided both finish the task, but both retain finger
  contact.
- Interpretation: disabling the hand shell removes an execution blocker, but
  does not make the route physically safe and does not reveal a guidance
  benefit.
- Formal and rendered labels match in all four panels.

## 2. D2 episode 0: disabling the hand is insufficient

[`episode_000_2x2.mp4`](d2_replay/episode_000_2x2.mp4)

- All four conditions collide and fail.
- Full physics contains hand and finger contact. With the hand shell disabled,
  finger, arm, and carried-target contacts remain.
- Guidance is active but does not produce a bypass route.
- The overall task/collision result and core contact pattern match the formal
  run. The full-physics guided replay adds a target-contact label that was not
  present in its formal rollout.

## 3. D2 episode 35: rare clean guidance positive

[`episode_035_2x2.mp4`](d2_positive_replay/episode_035_2x2.mp4)

- All four conditions finish the task.
- Both baselines contact the obstacle with the carried Can.
- Both guided rollouts remove that target contact and become safe successes.
- This shows that the XY point guidance can occasionally make the right local
  correction. Formal and rendered labels match in all four panels, but this is
  one positive transition rather than the aggregate result.

## 4. D3 episode 5: a formal positive that is not replay-stable

[`episode_005_2x2.mp4`](d3_replay/episode_005_2x2.mp4)

- The formal run recorded target-contact clearance under guidance.
- In this same-state, same-policy-seed CUDA replay, both guided panels contact
  the target again, although all four panels still complete the task.
- This video is retained to make the replay instability explicit; it is not
  presented as a successful guidance visualization.

## Replay caveat

Initial simulator state hashes and policy seeds are identical to the formal
evaluation, but CUDA inference and MuJoCo execution are not bitwise deterministic
across independent processes. Each directory's `manifest.json` stores both the
formal labels and the actual replay labels, including every mismatch. The
formal statistics remain those in
`../formal_seed812_n50_summary/report.md`, not statistics recomputed from these
visualization replays.
