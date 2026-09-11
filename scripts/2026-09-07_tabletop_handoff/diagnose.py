"""Bounded Tabletop oracle-route / source-handoff diagnostic (no training).

Replay executed source prefixes instead of cloning incomplete controller state.
All conditions retain full physics, actual observation history and H600 budget.
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from robosuite.utils import transform_utils as T

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "handoff_paired", ROOT / "scripts/2026-08-05_lan_o3dp_paper_aligned/paired_eval.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
C = P.rollout_common


def safe_json(x):
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def dump(path, x):
    path.write_text(json.dumps(safe_json(x), indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fingerprint(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class Runner:
    def __init__(self, args):
        self.args = args
        self.policy, ckpt = P.paper_guided_policy_from_checkpoint(
            ckpt_path=str(args.checkpoint), device=torch.device("cuda"))
        self.env, _ = P.FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt, env_name="PickPlaceCanTabletopRaisedMilk",
            render=False, render_offscreen=True, verbose=False)
        self.raw = C.raw_env_from_wrapper(self.env)
        self.bank = P.load_state_bank(args.state_bank)
        self.policy.start_episode()
        self.horizon = args.horizon

    def reset(self, episode):
        seed = 828000000 + episode
        seed_all(seed)
        self.raw.rng = np.random.default_rng(828000 + episode)
        record = self.bank["states"][f"PickPlaceCanTabletopRaisedMilk:828:{episode}"]
        self.obs = self.env.reset_to(record["state"])
        self.reset_metadata_audit = {
            "goal_xy_before_sync": np.asarray(self.raw.goal_xy).copy(),
            "goal_tray_xy": self.raw.sim.model.body_pos[self.raw.goal_tray_body_id, :2].copy(),
            "obstacle_xy_before_sync": np.asarray(self.raw.obstacle_xy).copy(),
            "obstacle_body_xy": self.raw.sim.model.body_pos[self.raw.obstacle_body_id, :2].copy()}
        self.raw.goal_xy = self.raw.sim.model.body_pos[self.raw.goal_tray_body_id, :2].copy()
        self.raw.obstacle_xy = self.raw.sim.model.body_pos[self.raw.obstacle_body_id, :2].copy()
        self.policy.start_episode()
        self.records = C.active_object_records(self.raw)
        self.maps = C.contact_maps(self.raw, self.records, "Can")
        self.categories = P.categorized_robot_geom_ids(self.raw)
        self.obstacle_ids = P.obstacle_geom_ids(self.raw, self.records, "Can")
        self.step = 0
        self.trace = []
        self.actions = []
        self.collision = False
        self.first_collision = None
        self.success = C.task_success(self.env)
        self.writer = None
        self.record("initial", None)

    def state(self):
        return self.raw.sim.get_state().flatten().copy()

    def eef(self):
        return C.current_raw_eef_position(self.raw).copy()

    def can(self):
        return self.raw.sim.data.body_xpos[self.raw.can_body_id].copy()

    def record(self, mode, action):
        contacts = P.categorized_robot_obstacle_contact_snapshot(self.raw, self.maps, self.categories)
        target_contacts, names = P.target_obstacle_contact_snapshot(self.raw, self.maps)
        contact = any(int(v) > 0 for v in contacts[:4]) or target_contacts > 0
        if contact and self.first_collision is None:
            self.first_collision = self.step
        self.collision |= contact
        self.trace.append({"step": self.step, "mode": mode,
                           "eef": self.eef(), "can": self.can(),
                           "goal_distance": float(np.linalg.norm(self.can()[:2] - self.raw.goal_xy)),
                           "grasp": bool(self.raw._check_grasp(self.raw.robots[0].gripper, self.raw.can)),
                           "contact": bool(contact), "success": bool(self.success),
                           "robot_contact_geoms": list(contacts[-1]),
                           "target_contact_objects": list(names)})

    def execute(self, action, mode):
        self.obs, _, done, _ = self.env.step(np.asarray(action, dtype=np.float32))
        self.step += 1
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.success = bool(C.task_success(self.env))
        self.record(mode, action)
        if self.writer is not None and self.step % 5 == 0:
            self.writer.append_data(self.env.render(mode="rgb_array", camera_name="agentview", height=320, width=320))
        return self.step < self.horizon and not self.success and not done

    def chunk(self, guided=False):
        # Both baseline and guided render obstacle geometry, matching the old protocol.
        cloud = P.visible_obstacle_pointcloud(self.raw, self.obstacle_ids)
        context = None
        if guided and len(cloud) and self.can()[2] > self.raw.table_offset[2] + .08:
            context = P.paper_lan_guidance_context_from_rollout_policy(
                self.policy, current_eef_pos=self.eef(), obstacle_points=cloud,
                guidance_scale=45., safety_distance=.08, xy_only=True,
                obstacle_cost_type="huber_hinge", obstacle_cost_smoothing=.01,
                tangent_ratio=0.)
        chunk, diagnostics = P.sample_raw_chunk(self.policy, self.obs, context)
        return chunk

    def capture(self, episode):
        self.reset(episode)
        captured = None
        while self.step < self.horizon and not self.success:
            if captured is None and not self.collision and self.step > 0 and self.can()[2] > self.raw.table_offset[2] + .08 and self.trace[-1]["grasp"]:
                captured = {"episode": episode, "step": self.step,
                            "prefix": np.asarray(self.actions).copy(), "state": self.state(),
                            "obs": {k: np.asarray(v).copy() for k, v in self.obs.items()},
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
                print(json.dumps({"captured": episode, "step": self.step}), flush=True)
            for action in self.chunk():
                if not self.execute(action, "source"):
                    break
        metrics = self.metrics()
        if captured is not None:
            captured["screen_metrics"] = metrics
        return captured, metrics

    def restore_prefix(self, saved):
        self.reset(saved["episode"])
        for action in saved["prefix"]:
            self.execute(action, "prefix")
        audit = {"state_max_abs": float(np.max(np.abs(self.state() - saved["state"]))),
                 "obs_max_abs": {k: float(np.max(np.abs(np.asarray(self.obs[k]) - v)))
                                 for k, v in saved["obs"].items()},
                 "state_hash": fingerprint(self.state()),
                 "obs_hash": {k: fingerprint(v) for k, v in self.obs.items()},
                 "prefix_collision": self.collision}
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        return audit

    def route(self):
        start = self.eef()
        can = self.can()
        goal = self.raw.goal_xy.copy()
        direction = goal - can[:2]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        obstacle = self.raw.sim.data.body_xpos[self.raw.obstacle_body_id, :2].copy()
        side = -1. if np.dot(obstacle - can[:2], normal) >= 0 else 1.
        # Conservative route construction proxy only; physical contacts are the gate.
        radius = float(self.raw.obstacle.horizontal_radius) + .10
        relative = obstacle - can[:2]
        advance = float(relative @ direction)
        lateral = float(relative @ normal) + side * radius
        offset = start[:2] - can[:2]
        via1 = start.copy()
        via1[:2] = can[:2] + lateral * normal + offset
        via2 = start.copy()
        via2[:2] = can[:2] + (advance + radius) * direction + lateral * normal + offset
        above = start.copy()
        above[:2] = goal + offset
        # Keep current carried height. Carrying rotation is preserved throughout.
        site = self.raw.robots[0].eef_site_id[self.raw.robots[0].arms[0]]
        rot = T.quat2axisangle(T.mat2quat(self.raw.sim.data.site_xmat[site].reshape(3, 3)))
        stages = [("bypass_side", via1, 1., 4), ("bypass_past", via2, 1., 4),
                  ("above_goal", above, 1., 4)]
        place = above.copy()
        place[2] = .865
        retreat = place.copy()
        retreat[2] += .14
        stages += [("lower", place, 1., 6), ("release", place, -1., 20),
                   ("retreat", retreat, -1., 4)]
        actions, phases = [], []
        previous = start.copy()
        for name, target, grip, dwell in stages:
            count = max(1, int(np.ceil(1.875 * np.linalg.norm(target - previous) / .006)))
            u = np.linspace(1. / count, 1., count)
            blend = 10*u**3 - 15*u**4 + 6*u**5
            positions = previous + blend[:, None] * (target - previous)
            for position in list(positions) + [target] * dwell:
                actions.append(np.r_[position, rot, grip].astype(np.float32))
                phases.append(name)
            previous = target
        return {"actions": np.asarray(actions), "phases": phases, "direction": direction,
                "obstacle": obstacle, "radius_proxy": radius, "side": side,
                "stages": stages}

    def passed_geometry(self, direction):
        """Exact mesh / primitive support along the start-to-goal axis.

        This only checks the past-obstacle handoff plane, not whole-arm safety.
        Physical contact metrics still include the arm and entire prefix.
        """
        axis = np.r_[direction, 0.]
        model, data = self.raw.sim.model, self.raw.sim.data
        def interval(geom):
            center = float(data.geom_xpos[geom] @ axis)
            local = data.geom_xmat[geom].reshape(3, 3).T @ axis
            kind, size = int(model.geom_type[geom]), model.geom_size[geom]
            if kind == 7:  # MuJoCo mesh vertices are in the compiled geom frame.
                mesh = int(model.geom_dataid[geom])
                adr, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                projection = model.mesh_vert[adr:adr + count] @ local
                return center + float(projection.min()), center + float(projection.max())
            if kind == 6:  # box
                radius = float(np.abs(local) @ size)
            elif kind == 2:  # sphere
                radius = float(size[0])
            elif kind == 3:  # capsule
                radius = float(size[0] + size[1] * abs(local[2]))
            elif kind == 5:  # cylinder
                radius = float(size[0] * np.linalg.norm(local[:2]) + size[1] * abs(local[2]))
            elif kind == 4:  # ellipsoid
                radius = float(np.linalg.norm(size * local))
            else:
                raise ValueError(f"Unsupported collision geom type {kind}")
            return center - radius, center + radius
        actor_ids = set(self.categories[1]) | set(self.maps[1])
        obstacle_ids = set(self.maps[2])
        actor_min = min(interval(g)[0] for g in actor_ids)
        obstacle_max = max(interval(g)[1] for g in obstacle_ids)
        return {"actor_min_projection": actor_min, "obstacle_max_projection": obstacle_max,
                "past_plane_margin_m": actor_min - obstacle_max,
                "whole_gripper_and_can_past": bool(actor_min > obstacle_max)}

    def metrics(self):
        return {"reset_metadata_audit": self.reset_metadata_audit,
                "success": bool(self.success), "safe_success": bool(self.success and not self.collision),
                "collision_any": bool(self.collision), "ncr": bool(not self.success and not self.collision),
                "success_with_collision": bool(self.success and self.collision),
                "collision_failure": bool(not self.success and self.collision),
                "steps": self.step, "first_collision_step": self.first_collision,
                "final_goal_distance": self.trace[-1]["goal_distance"]}

    def branch(self, saved, condition, repeat, out):
        audit = self.restore_prefix(saved)
        dump(out / f"ep{saved['episode']:02d}_{condition}_r{repeat}_restore.json", audit)
        if audit["state_max_abs"] > 1e-6 or audit["obs_max_abs"]["agent_pos"] > 1e-5 or audit["prefix_collision"]:
            raise RuntimeError("Shared physical prefix did not reproduce; inspect saved restore audit before branching")
        route = saved["route"]
        handoff = None
        first_chunk = None
        # Video is captured within the evaluated branch, never by a later replay.
        if repeat == 0:
            self.writer = imageio.get_writer(out / f"ep{saved['episode']:02d}_{condition}.mp4", fps=4)
        try:
            if condition in ("short", "long", "teacher", "near_goal"):
                limit = 8 if condition == "short" else len(route["actions"])
                for index, (action, phase) in enumerate(zip(route["actions"], route["phases"])):
                    if index >= limit:
                        break
                    if condition == "long" and phase not in ("bypass_side", "bypass_past"):
                        break
                    if condition == "near_goal" and phase not in ("bypass_side", "bypass_past", "above_goal"):
                        break
                    if not self.execute(action, phase):
                        break
            if condition != "teacher" and not self.success and self.step < self.horizon:
                handoff = {"step": self.step, "eef": self.eef(), "can": self.can(),
                           "collision_before_handoff": self.collision,
                           "grasp": self.trace[-1]["grasp"],
                           "geometry": self.passed_geometry(route["direction"]),
                           "actor_past_obstacle_proxy": float(min(self.eef()[:2] @ route["direction"], self.can()[:2] @ route["direction"]) - route["obstacle"] @ route["direction"])}
                # sample_raw_chunk has no action queue; env.step retained real frame history.
                while self.step < self.horizon and not self.success:
                    chunk = self.chunk(condition == "guided")
                    if first_chunk is None:
                        first_chunk = chunk.copy()
                    for action in chunk:
                        if not self.execute(action, condition):
                            break
            result = {"episode": saved["episode"], "condition": condition, "repeat": repeat,
                      "start_step": saved["step"], "restore_audit": audit,
                      "handoff": handoff, "first_source_chunk": first_chunk, **self.metrics()}
            dump(out / f"ep{saved['episode']:02d}_{condition}_r{repeat}_trace.json", self.trace)
            return result
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None


def main():
    parser = argparse.ArgumentParser()
    manifest = json.loads((ROOT / "outputs/tabletop_wide_support/raised_milk_paired_seed828_n50/baseline/manifest.json").read_text())
    parser.add_argument("--checkpoint", type=Path, default=Path(manifest["checkpoint"]))
    parser.add_argument("--state-bank", type=Path, default=Path(manifest["state_bank"]))
    parser.add_argument("--episodes", type=int, nargs="+", default=[8, 17, 25, 37, 43, 48, 49])
    parser.add_argument("--max-states", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", choices=("teacher", "baseline", "guided", "short", "long", "near_goal"),
                        default=["teacher", "baseline", "guided", "short", "long"])
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old_manifest = args.out_dir / "manifest.json"
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        for key in ("checkpoint", "state_bank", "horizon"):
            if str(old[key]) != str(getattr(args, key)):
                raise ValueError(f"Cannot resume with a changed {key}; use a new output directory")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dump(args.out_dir / "manifest.json", {**vars(args), "checkpoint": str(args.checkpoint),
         "state_bank": str(args.state_bank), "out_dir": str(args.out_dir),
         "selection": "ascending archived baseline collision failures; retain all screening outcomes",
         "guidance": {"rho": 45, "Q": .08, "smoothing": .01, "phase": "target_lifted"}})
    runner = Runner(args)
    rows_path = args.out_dir / "results.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    completed = {(r["episode"], r["condition"], r["repeat"]) for r in rows}
    selected = 0
    try:
        for episode in args.episodes:
            if selected >= args.max_states:
                break
            cache = args.out_dir / f"ep{episode:02d}_capture.pkl"
            screen_path = args.out_dir / f"ep{episode:02d}_screen.json"
            if not cache.exists() and screen_path.exists() and not json.loads(screen_path.read_text())["collision_failure"]:
                continue
            if cache.exists():
                with cache.open("rb") as f:
                    saved = pickle.load(f)
            else:
                saved, screen = runner.capture(episode)
                dump(args.out_dir / f"ep{episode:02d}_screen.json", screen)
                dump(args.out_dir / f"ep{episode:02d}_screen_trace.json", runner.trace)
                print(json.dumps(safe_json({"screen": episode, **screen})), flush=True)
                if saved is None or not screen["collision_failure"]:
                    continue
                runner.restore_prefix(saved)
                saved["route"] = runner.route()
                with cache.open("wb") as f:
                    pickle.dump(saved, f)
                dump(args.out_dir / f"ep{episode:02d}_route.json", saved["route"])
            selected += 1
            for repeat in range(args.repeats):
                for condition in args.conditions:
                    key = (episode, condition, repeat)
                    if condition != "teacher":
                        reference = [r for r in rows if r["episode"] == episode and r["condition"] == "teacher" and r["repeat"] == repeat]
                        if not reference or not reference[0]["safe_success"]:
                            raise RuntimeError("A safe full-teacher gate is required for this episode and repetition")
                    if key in completed:
                        continue
                    result = runner.branch(saved, condition, repeat, args.out_dir)
                    with rows_path.open("a") as f:
                        f.write(json.dumps(safe_json(result)) + "\n")
                    rows.append(safe_json(result))
                    print(json.dumps(safe_json({k: v for k, v in result.items() if k not in ("first_source_chunk", "restore_audit", "handoff")})), flush=True)
                    if condition == "teacher" and not result["safe_success"]:
                        print("Teacher gate failed; stopping for diagnosis before interpreting source handoffs.", flush=True)
                        return
        dump(args.out_dir / "summary.json", {"selected_states": selected, "rows": len(rows),
             "conditions": {condition: {"n": len(rs), **{name: float(np.mean([r[name] for r in rs])) for name in ("success", "safe_success", "collision_any", "ncr")}}
                            for condition in sorted({r["condition"] for r in rows})
                            if (rs := [r for r in rows if r["condition"] == condition])}})
    finally:
        runner.raw.close()


if __name__ == "__main__":
    main()
