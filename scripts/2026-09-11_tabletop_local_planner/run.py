"""Local planner evaluation with inclusive success and in-execution MP4s."""

import argparse, importlib.util, json, random, time, os
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from runtime import LocalPlanner

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "r", ROOT / "scripts/2026-09-09_tabletop_guidance_reeval/run.py"
)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
P, C = R.P, R.C


def rng_get():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )


def rng_set(s):
    torch.set_rng_state(s[0])
    torch.cuda.set_rng_state(s[1])
    np.random.set_state(s[2])
    random.setstate(s[3])


class Eval(R.Evaluation):
    def run(self, ep, cond, folder):
        folder.mkdir(parents=True, exist_ok=False)
        rec = self.bank["states"][f"PickPlaceCanTabletopRaisedMilk:828:{ep}"]
        seed = 828000000 + ep
        self.env.layout_seed = rec["layout_seed"]
        R.seed(seed)
        self.env.reset_to(rec["state"])
        R.seed(seed)
        self.policy.start_episode()
        obs = self.env.reset_to(rec["state"])
        assert self.env.reset_audit["restored_state_hash"] == rec["state_hash"]
        P.configure_hand_collision(self.raw, enabled=True)
        records = C.active_object_records(self.raw)
        maps = C.contact_maps(self.raw, records, "Can")
        cats = P.categorized_robot_geom_ids(self.raw)
        ids = P.obstacle_geom_ids(self.raw, records, "Can")
        model = self.raw.sim.model
        grip = {
            i
            for i in range(model.ngeom)
            if (model.geom_id2name(i) or "").startswith("gripper")
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        }
        target = set(maps[1])
        assert target and grip
        planner = LocalPlanner(self.raw, ids) if cond == "planner" else None
        trace = []
        actions = []
        states = []
        chunks = []
        observations = []
        source_chunks = []
        guided_chunks = []
        output_chunks = []
        clouds = []
        vsteps = []
        collision = False
        first_collision = None
        hold_start = None
        success = False
        step = 0
        first_strict = None
        video = None
        started = time.monotonic()

        def record():
            nonlocal collision, first_collision, hold_start, success, first_strict
            robot = P.categorized_robot_obstacle_contact_snapshot(self.raw, maps, cats)
            tc, tn = P.target_obstacle_contact_snapshot(self.raw, maps)
            contact = any(int(v) > 0 for v in robot[:4]) or tc > 0
            if contact and first_collision is None:
                first_collision = step
            collision |= contact
            touches = set()
            for con in self.raw.sim.data.contact[: self.raw.sim.data.ncon]:
                a, b = int(con.geom1), int(con.geom2)
                if a in target and b in grip:
                    touches.add(b)
                if b in target and a in grip:
                    touches.add(a)
            can = self.raw.sim.data.body_xpos[self.raw.can_body_id].copy()
            eef = C.current_raw_eef_position(self.raw)
            goal = self.raw.goal_xy.copy()
            assert np.array_equal(goal, model.body_pos[self.raw.goal_tray_body_id, :2])
            inside = bool(
                np.all(np.abs(can[:2] - goal) < self.raw.goal_tray_inner_size / 2)
                and 0.8 < can[2] < 0.9
            )
            eligible = inside and not touches
            if eligible:
                if hold_start is None:
                    hold_start = step
            else:
                hold_start = None
            duration = (step - hold_start) / 20 if hold_start is not None else 0.0
            strict = bool(C.task_success(self.env))
            success = bool(strict or duration >= 0.5)
            if strict and first_strict is None:
                first_strict = step
            trace.append(
                {
                    "step": step,
                    "can": can,
                    "eef": eef,
                    "goal_xy": goal,
                    "inside_goal": inside,
                    "no_gripper_contact": not touches,
                    "hold_seconds": duration,
                    "gripper_contact_geoms": [
                        model.geom_id2name(i) for i in sorted(touches)
                    ],
                    "grasp": bool(
                        self.raw._check_grasp(self.raw.robots[0].gripper, self.raw.can)
                    ),
                    "contact": bool(contact),
                    "collision_any": bool(collision),
                    "robot_contact_geoms": list(robot[-1]),
                    "target_contact_objects": list(tn),
                    "success": bool(success),
                    "strict_success": strict,
                    "height_gate": bool(can[2] > 0.88),
                    "obs_hashes": {k: R.sha_array(v) for k, v in obs.items()},
                }
            )
            states.append(self.raw.sim.get_state().flatten().copy())
            if video is not None and (step % 2 == 0 or success or step == 600):
                pixels = self.env.render(
                    mode="rgb_array", camera_name="agentview", height=384, width=384
                )
                im = Image.new("RGB", (384, 432), "black")
                im.paste(Image.fromarray(np.asarray(pixels)), (0, 48))
                draw = ImageDraw.Draw(im)
                draw.text(
                    (7, 4), f"Layout {ep:02d} | {cond} | step {step}/600", fill="white"
                )
                draw.text(
                    (7, 25),
                    f"Placed {int(success)} | Contact {int(collision)} | Hold {duration:.2f}s",
                    fill="white",
                )
                video.append_data(np.asarray(im))
                vsteps.append(step)

        video = imageio.get_writer(
            str(folder / "rollout.partial.mp4"), fps=10, codec="libx264", quality=7
        )
        try:
            record()
            while step < 600 and not success:
                cloud = P.visible_obstacle_pointcloud(self.raw, ids)
                enabled = (
                    cond == "guided"
                    and len(cloud) > 0
                    and P.guidance_phase_active(
                        self.raw, records, "Can", "target_lifted"
                    )
                )
                context = None
                if enabled:
                    context = P.paper_lan_guidance_context_from_rollout_policy(
                        self.policy,
                        current_eef_pos=np.asarray(
                            C.latest_obs_value(obs, "agent_pos")[:3]
                        ),
                        obstacle_points=cloud,
                        guidance_scale=45.0,
                        safety_distance=0.08,
                        xy_only=True,
                        obstacle_cost_type="huber_hinge",
                        obstacle_cost_smoothing=0.01,
                        tangent_ratio=0.0,
                    )
                obs_copy = {k: np.asarray(v).copy() for k, v in obs.items()}
                state = rng_get()
                output, diags = P.sample_raw_chunk(self.policy, obs, context)
                source = output.copy()
                guided = output.copy()
                planner_info = None
                if planner is not None:
                    output, planner_info = planner.control(source)
                chunks.append(
                    {
                        "step": step,
                        "enabled": bool(enabled),
                        "cpu_rng_hash": R.sha_array(state[0].numpy()),
                        "cuda_rng_hash": R.sha_array(state[1].numpy()),
                        "guidance_diagnostics": diags,
                        "planner": planner_info,
                    }
                )
                observations.append(obs_copy)
                clouds.append(cloud.copy())
                source_chunks.append(source)
                guided_chunks.append(guided)
                output_chunks.append(output.copy())
                for action in output:
                    obs, _, done, _ = self.env.step(action)
                    actions.append(action.copy())
                    step += 1
                    record()
                    if success or step >= 600 or done:
                        break
                if done and not success and step < 600:
                    raise RuntimeError(
                        "Unexpected environment termination before placement/H600"
                    )
        finally:
            video.close()
        R.dump(folder / "trace.json", trace)
        R.dump(folder / "chunks.json", chunks)
        R.dump(folder / "reset.json", self.env.reset_audit)
        np.savez_compressed(
            folder / "trajectory.npz",
            actions=np.asarray(actions),
            states=np.asarray(states),
            video_steps=np.asarray(vsteps),
            source_chunks=np.asarray(source_chunks),
            guided_chunks=np.asarray(guided_chunks),
            output_chunks=np.asarray(output_chunks),
        )
        od = {"initial_" + k: v for k, v in self.env.initial_obs.items()}
        for k in observations[0]:
            od["chunks_" + k] = np.stack([o[k] for o in observations])
        od.update({f"obstacle_cloud_{i:03d}": v for i, v in enumerate(clouds)})
        np.savez_compressed(folder / "observations.npz", **od)
        (folder / "rollout.partial.mp4").replace(folder / "rollout.mp4")
        result = {
            "episode": ep,
            "condition": cond,
            "steps": step,
            "success": bool(success),
            "safe_success": bool(success and not collision),
            "collision_any": bool(collision),
            "success_with_collision": bool(success and collision),
            "collision_failure": bool(not success and collision),
            "ncr": bool(not success and not collision),
            "first_collision_step": first_collision,
            "first_strict_success_step": first_strict,
            "gate_active_chunks": sum(c["enabled"] for c in chunks),
            "planner_changed_chunks": sum(
                bool(c["planner"] and c["planner"]["changed"]) for c in chunks
            ),
            "planner_triggers": planner.triggers if planner is not None else 0,
            "initial_state_hash": rec["state_hash"],
            "reset_audit": self.env.reset_audit,
            "wall_seconds": time.monotonic() - started,
            "video_frames": len(vsteps),
            "folder": str(folder.relative_to(self.cli.out)),
        }
        R.dump(folder / "result.json", result)
        print(
            json.dumps(R.safe({k: v for k, v in result.items() if k != "reset_audit"})),
            flush=True,
        )
        return result


def compare(a, b, repeat=False):
    ra = json.loads((a / "result.json").read_text())
    rb = json.loads((b / "result.json").read_text())
    assert (
        ra["initial_state_hash"] == rb["initial_state_hash"]
        and ra["reset_audit"]["initial_obs_hashes"]
        == rb["reset_audit"]["initial_obs_hashes"]
    )
    if repeat:
        assert json.loads((a / "trace.json").read_text()) == json.loads(
            (b / "trace.json").read_text()
        )
        with np.load(a / "trajectory.npz") as x, np.load(b / "trajectory.npz") as y:
            assert np.array_equal(x["actions"], y["actions"]) and np.array_equal(
                x["states"], y["states"]
            )
    return {"initial_observations_equal": True, "repeat_trajectory_equal": repeat}


def main():
    default = json.loads(
        (
            ROOT
            / "outputs/tabletop_wide_support/raised_milk_paired_seed828_n50/baseline/manifest.json"
        ).read_text()
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path(default["checkpoint"]))
    ap.add_argument("--state-bank", type=Path, default=Path(default["state_bank"]))
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument(
        "--conditions",
        nargs="+",
        choices=["baseline", "guided", "planner"],
        default=["baseline", "planner"],
    )
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.repeats < 1 or len(set(args.episodes)) != len(args.episodes):
        raise ValueError("Invalid episodes/repeats")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=False)
    R.dump(
        args.out / "manifest.json",
        {
            **vars(args),
            "source_hashes": {
                p.name: R.sha_file(p) for p in Path(__file__).parent.glob("*.py")
            },
            "checkpoint_sha256": R.sha_file(args.checkpoint),
            "state_bank_sha256": R.sha_file(args.state_bank),
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "success": "old strict OR released placement held0.5s",
            "geometry": "oracle compiled obstacle+support and actor MVEE; no pointcloud changes",
            "planner": {
                "margin_m": 0.01,
                "fit_padding_m": 0.002,
                "max_target_increment_m": 0.006,
            },
        },
    )
    ev = Eval(args)
    try:
        # Validate the checkpoint's physical absolute-action contract before execution.
        stats = ev.policy.action_normalization_stats
        assert len(stats) == 1 and "abs" in next(iter(stats)), str(stats.keys())
        for ep in args.episodes:
            for rep in range(args.repeats):
                for cond in args.conditions:
                    ev.run(ep, cond, args.out / f"ep{ep:02d}" / f"r{rep}" / cond)
                base = args.out / f"ep{ep:02d}" / f"r{rep}" / args.conditions[0]
                R.dump(
                    base.parent / "pair_audit.json",
                    {c: compare(base, base.parent / c) for c in args.conditions[1:]},
                )
            if args.repeats > 1:
                R.dump(
                    args.out / f"ep{ep:02d}" / "repeat_audit.json",
                    {
                        f"{c}_r{rep}": compare(
                            args.out / f"ep{ep:02d}/r0" / c,
                            args.out / f"ep{ep:02d}/r{rep}" / c,
                            True,
                        )
                        for c in args.conditions
                        for rep in range(1, args.repeats)
                    },
                )
        R.dump(
            args.out / "complete.json",
            {
                "episodes": args.episodes,
                "conditions": args.conditions,
                "repeats": args.repeats,
            },
        )
    finally:
        ev.raw.close()


if __name__ == "__main__":
    main()
