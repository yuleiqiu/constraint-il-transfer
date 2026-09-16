"""Paired oracle-mask versus RGB-D-perceived frozen-policy rollouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from experiment_common import (
    CONDITIONS,
    DEFAULT_CHECKPOINT,
    ENVIRONMENTS,
    PILOT_EPISODES,
    CorrectedResetEnv,
    audit_absolute_eef_actions,
    dump_json,
    json_safe,
    load_state_bank,
    sha256_file,
    state_hash,
)
from models import load_model_bundle
from oracle_audit import (
    audit_perception_frame,
    geom_ids_by_role,
    render_oracle_masks,
)
from perception import PerceptionFrame, PerceptionProvider, ProviderConfig, array_sha256
from runtime_io import (
    RgbdSensor,
    preserve_global_rng,
    replace_policy_pointcloud,
)
from weights import force_offline_runtime, verify_manifest


ROOT = Path(__file__).resolve().parents[2]
PAIRED_EVAL_PATH = (
    ROOT / "scripts/2026-08-05_lan_o3dp_paper_aligned/paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("perception_paired_eval", PAIRED_EVAL_PATH)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
C = P.rollout_common


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_hashes() -> dict[str, str]:
    result = {
        "torch_cpu": array_sha256(torch.get_rng_state().cpu().numpy()),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode("utf-8")).hexdigest(),
        "python": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
    }
    for index, state in enumerate(torch.cuda.get_rng_state_all()):
        result[f"torch_cuda_{index}"] = array_sha256(state.cpu().numpy())
    return result


def label_visualization(frame: PerceptionFrame | None, shape=(384, 384)) -> Image.Image:
    colors = {
        "target": np.array([235, 55, 55], dtype=np.uint8),
        "goal": np.array([40, 210, 70], dtype=np.uint8),
        "milk": np.array([100, 160, 255], dtype=np.uint8),
        "support": np.array([240, 180, 45], dtype=np.uint8),
    }
    if frame is None or not frame.masks:
        return Image.new("RGB", shape, "black")
    first_mask = next(iter(frame.masks.values()))
    visualization = np.zeros((*np.asarray(first_mask).shape, 3), dtype=np.uint8)
    for role, color in colors.items():
        if role in frame.masks:
            visualization[np.asarray(frame.masks[role], dtype=bool)] = color
    return Image.fromarray(visualization).resize(shape, resample=Image.Resampling.NEAREST)


def gripper_target_contact_geoms(raw_env, target_ids: set[int], gripper_ids: set[int]) -> list[str]:
    touched = set()
    for contact in raw_env.sim.data.contact[: raw_env.sim.data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in target_ids and geom2 in gripper_ids:
            touched.add(geom2)
        if geom2 in target_ids and geom1 in gripper_ids:
            touched.add(geom1)
    return [raw_env.sim.model.geom_id2name(index) for index in sorted(touched)]


class Evaluation:
    def __init__(self, cli) -> None:
        self.cli = cli
        self.environment = ENVIRONMENTS[cli.environment]
        self.bank = load_state_bank(cli.state_bank)
        self.policy, checkpoint = P.paper_guided_policy_from_checkpoint(
            ckpt_path=str(cli.checkpoint), device=torch.device("cuda")
        )
        env, _ = P.FileUtils.env_from_checkpoint(
            ckpt_dict=checkpoint,
            env_name=self.environment["name"],
            render=False,
            render_offscreen=True,
            verbose=False,
        )
        self.env = CorrectedResetEnv(env)
        self.raw = self.env.raw
        self.bundle = load_model_bundle(cli.weights, device="cuda")
        self.locked = json.loads(cli.locked_config.read_text())
        self.provider_config = ProviderConfig(
            prompts=self.locked["prompts"],
            scene_has_obstacle=self.environment["has_obstacle"],
            box_threshold=self.locked["box_threshold"],
            text_threshold=self.locked["text_threshold"],
        )
        stats = self.policy.action_normalization_stats
        if len(stats) != 1 or "abs" not in next(iter(stats)):
            raise RuntimeError(
                f"checkpoint is not the validated absolute-EEF contract: {stats.keys()}"
            )
        self.role_geom_ids = None

    def _record_for_episode(self, episode: int) -> dict[str, Any]:
        seed = int(self.environment["seed"])
        env_name = self.environment["name"]
        key = f"{env_name}:{seed}:{episode}"
        if key not in self.bank["states"]:
            raise KeyError(f"state bank does not contain {key}")
        record = self.bank["states"][key]
        if record["layout_seed"] != seed * 1000 + episode:
            raise RuntimeError(f"layout seed mismatch: {key}")
        if state_hash(record["state"]) != record["state_hash"]:
            raise RuntimeError(f"state hash mismatch: {key}")
        return record

    def _reset(self, record, condition: str):
        # The perceived condition disables the inner oracle provider before
        # reset.  task_pointcloud is then absent until the outer RGB-D provider
        # injects its own two-frame history.
        oracle_enabled = condition == "oracle_mask_source"
        self.env.base_env.target_pointcloud_config["enabled"] = oracle_enabled
        self.env.layout_seed = record["layout_seed"]
        policy_seed = int(self.environment["seed"]) * 1_000_000 + self.cli.current_episode
        seed_everything(policy_seed)
        self.env.reset_to(record["state"])
        seed_everything(policy_seed)
        self.policy.start_episode()
        observation = self.env.reset_to(record["state"])
        if state_hash(self.env.get_state()) != record["state_hash"]:
            raise RuntimeError("restored simulator state does not match bank")
        if oracle_enabled and "task_pointcloud" not in observation:
            raise RuntimeError("oracle source did not produce task_pointcloud")
        if not oracle_enabled and "task_pointcloud" in observation:
            raise RuntimeError("perceived path executed the oracle point-cloud provider")
        return observation, policy_seed

    def run_condition(self, episode: int, condition: str, folder: Path) -> dict[str, Any]:
        folder.mkdir(parents=True, exist_ok=False)
        self.cli.current_episode = episode
        record = self._record_for_episode(episode)
        observation, policy_seed = self._reset(record, condition)
        P.configure_hand_collision(self.raw, enabled=True)
        sensor = RgbdSensor(self.raw)
        self.role_geom_ids = geom_ids_by_role(self.raw)
        provider = None
        perceived: PerceptionFrame | None = None
        with preserve_global_rng():
            initial_rgbd = sensor.capture()
        native_initial_hashes = {
            key: array_sha256(value) for key, value in observation.items()
        }
        initial_rng_before_perception = rng_hashes()
        perception_failure = False
        perception_failure_step = None
        if condition == "perceived_mask_source":
            provider = PerceptionProvider(
                detector=self.bundle.detector,
                segmenter=self.bundle.segmenter,
                tracker_factory=self.bundle.tracker_factory,
                config=self.provider_config,
            )
            with preserve_global_rng():
                perceived = provider.start_episode(initial_rgbd)
            observation = replace_policy_pointcloud(observation, provider)
            perception_failure = not perceived.valid
            if perception_failure:
                perception_failure_step = 0
        initial_rng_after_perception = rng_hashes()
        if initial_rng_before_perception != initial_rng_after_perception:
            raise RuntimeError("perception changed the frozen policy RNG stream")

        with preserve_global_rng():
            initial_oracle_masks = render_oracle_masks(
                self.raw,
                camera_name=sensor.camera_name,
                height=sensor.height,
                width=sensor.width,
                role_geom_ids=self.role_geom_ids,
            )
            initial_audit = (
                audit_perception_frame(perceived, initial_rgbd, initial_oracle_masks)
                if perceived is not None
                else None
            )
        initial_policy_hashes = {
            key: array_sha256(value) for key, value in observation.items()
        }
        np.savez_compressed(
            folder / "initial_rgbd_and_masks.npz",
            rgb=initial_rgbd.rgb,
            depth_m=initial_rgbd.depth_m,
            intrinsic=initial_rgbd.intrinsic,
            camera_pose=initial_rgbd.camera_pose,
            **{
                f"oracle_mask_{role}": mask
                for role, mask in initial_oracle_masks.items()
            },
            **(
                {
                    f"perceived_mask_{role}": mask
                    for role, mask in perceived.masks.items()
                }
                if perceived is not None
                else {}
            ),
        )
        initial_record = {
            "initial_state_hash": record["state_hash"],
            "policy_seed": policy_seed,
            "native_observation_hashes": native_initial_hashes,
            "policy_observation_hashes": initial_policy_hashes,
            "rgb_sha256": array_sha256(initial_rgbd.rgb),
            "depth_sha256": array_sha256(initial_rgbd.depth_m),
            "calibration_hashes": sensor.calibration_hashes(),
            "rng_before_perception": initial_rng_before_perception,
            "rng_after_perception": initial_rng_after_perception,
            "oracle_provider_enabled": condition == "oracle_mask_source",
            "perception_valid": None if perceived is None else perceived.valid,
            "reset_audit": self.env.reset_audit,
        }
        dump_json(folder / "initial_audit.json", initial_record)

        records = C.active_object_records(self.raw)
        contact_maps = C.contact_maps(self.raw, records, "Can")
        categories = P.categorized_robot_geom_ids(self.raw)
        target_ids = set(contact_maps[1])
        gripper_ids = set(categories[1])
        low, high = C.action_limits(self.env, self.policy.policy.ac_dim)
        trace = []
        actions = []
        states = []
        chunk_records = []
        chunk_clouds: dict[str, np.ndarray] = {}
        first_collision_step = None
        first_strict_success_step = None
        collision_any = False
        hold_start = None
        success = False
        done = False
        step = 0
        action_spec_exceedance_count = 0
        action_spec_exceedance_max = 0.0
        action_contract_violation_count = 0
        video_steps = []
        started = time.monotonic()
        diagnostics_stream = (folder / "perception_diagnostics.jsonl").open("w")
        writer = imageio.get_writer(
            str(folder / "rollout.partial.mp4"), fps=10, codec="libx264", quality=7
        )

        def score_and_record(current_audit=None) -> None:
            nonlocal collision_any, first_collision_step, hold_start, success
            nonlocal first_strict_success_step
            robot = P.categorized_robot_obstacle_contact_snapshot(
                self.raw, contact_maps, categories
            )
            target_contacts, target_names = P.target_obstacle_contact_snapshot(
                self.raw, contact_maps
            )
            contact = any(int(value) > 0 for value in robot[:4]) or target_contacts > 0
            collision_any = collision_any or contact
            if contact and first_collision_step is None:
                first_collision_step = step
            touched = gripper_target_contact_geoms(self.raw, target_ids, gripper_ids)
            can = self.raw.sim.data.body_xpos[self.raw.can_body_id].copy()
            goal = self.raw.sim.model.body_pos[self.raw.goal_tray_body_id, :2].copy()
            inside = bool(
                np.all(np.abs(can[:2] - goal) < self.raw.goal_tray_inner_size / 2.0)
                and self.raw.table_offset[2] < can[2] < self.raw.table_offset[2] + 0.1
            )
            released = inside and not touched
            if released:
                if hold_start is None:
                    hold_start = step
            else:
                hold_start = None
            held_seconds = (step - hold_start) / 20.0 if hold_start is not None else 0.0
            strict = bool(C.task_success(self.env))
            if strict and first_strict_success_step is None:
                first_strict_success_step = step
            success = bool(strict or held_seconds >= 0.5)
            trace.append(
                {
                    "step": step,
                    "can": can,
                    "goal_xy": goal,
                    "inside_goal": inside,
                    "released": released,
                    "hold_seconds": held_seconds,
                    "strict_success": strict,
                    "success": success,
                    "collision_any": collision_any,
                    "robot_contact_geoms": robot[-1],
                    "target_contact_objects": target_names,
                    "gripper_target_contact_geoms": touched,
                    "perception_valid": None if perceived is None else perceived.valid,
                    "policy_observation_hashes": {
                        key: array_sha256(value) for key, value in observation.items()
                    },
                }
            )
            states.append(self.raw.sim.get_state().flatten().copy())
            if step % 2 == 0 or success or perception_failure or step == self.cli.horizon:
                pixels = self.env.render(
                    mode="rgb_array", camera_name="agentview", height=384, width=384
                )
                canvas = Image.new("RGB", (768, 432), "black")
                canvas.paste(Image.fromarray(np.asarray(pixels, dtype=np.uint8)), (0, 48))
                canvas.paste(label_visualization(perceived), (384, 48))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (8, 5),
                    f"{self.cli.environment} ep{episode:02d} | {condition} | {step}/{self.cli.horizon}",
                    fill="white",
                )
                draw.text(
                    (8, 25),
                    f"Task {int(success)} | Contact {int(collision_any)} | Perception failure {int(perception_failure)}",
                    fill="white",
                )
                writer.append_data(np.asarray(canvas))
                video_steps.append(step)
            if perceived is not None:
                diagnostic = {
                    "step": step,
                    "perception": perceived.diagnostics,
                    "oracle_audit": (
                        None if current_audit is None else current_audit.diagnostics
                    ),
                }
                diagnostics_stream.write(json.dumps(json_safe(diagnostic)) + "\n")
                diagnostics_stream.flush()

        score_and_record(initial_audit)
        try:
            while (
                step < self.cli.horizon
                and not success
                and not perception_failure
                and not done
            ):
                before_policy_rng = rng_hashes()
                raw_chunk, _ = P.sample_raw_chunk(self.policy, observation, None)
                chunk_index = len(chunk_records)
                chunk_records.append(
                    {
                        "chunk": chunk_index,
                        "step": step,
                        "policy_rng_before": before_policy_rng,
                        "observation_hashes": {
                            key: array_sha256(value) for key, value in observation.items()
                        },
                    }
                )
                if perceived is not None:
                    chunk_clouds[f"chunk_{chunk_index:03d}_task"] = perceived.task_points
                    chunk_clouds[f"chunk_{chunk_index:03d}_target"] = perceived.target_points
                    chunk_clouds[f"chunk_{chunk_index:03d}_goal"] = perceived.goal_points
                    chunk_clouds[f"chunk_{chunk_index:03d}_obstacle"] = perceived.obstacle_points
                for action in raw_chunk:
                    if step >= self.cli.horizon:
                        break
                    action = np.asarray(action, dtype=np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise RuntimeError(
                            f"invalid absolute-EEF action at episode {episode}, step {step}"
                        )
                    contract = audit_absolute_eef_actions(action[None])
                    action_contract_violation_count += contract["violation_count"]
                    exceedance = np.maximum(action - high, low - action)
                    exceedance = np.maximum(exceedance, 0.0)
                    if np.any(exceedance > 1e-5):
                        action_spec_exceedance_count += int(
                            np.count_nonzero(exceedance > 1e-5)
                        )
                        action_spec_exceedance_max = max(
                            action_spec_exceedance_max, float(np.max(exceedance))
                        )
                    observation, _, done, _ = self.env.step(action)
                    actions.append(action.copy())
                    step += 1
                    current_audit = None
                    if provider is not None:
                        rng_before = rng_hashes()
                        with preserve_global_rng():
                            rgbd = sensor.capture()
                            perceived = provider.track(rgbd)
                        if rng_hashes() != rng_before:
                            raise RuntimeError(
                                f"perception changed policy RNG at step {step}"
                            )
                        observation = replace_policy_pointcloud(observation, provider)
                        perception_failure = not perceived.valid
                        if perception_failure and perception_failure_step is None:
                            perception_failure_step = step
                        with preserve_global_rng():
                            oracle_masks = render_oracle_masks(
                                self.raw,
                                camera_name=sensor.camera_name,
                                height=sensor.height,
                                width=sensor.width,
                                role_geom_ids=self.role_geom_ids,
                            )
                            current_audit = audit_perception_frame(
                                perceived, rgbd, oracle_masks
                            )
                    score_and_record(current_audit)
                    if success or perception_failure or done:
                        break
        finally:
            diagnostics_stream.close()
            writer.close()

        dump_json(folder / "trace.json", trace)
        dump_json(folder / "chunks.json", chunk_records)
        np.savez_compressed(
            folder / "trajectory.npz",
            actions=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
            states=np.asarray(states),
            video_steps=np.asarray(video_steps, dtype=np.int32),
        )
        np.savez_compressed(folder / "perceived_pointclouds_at_chunks.npz", **chunk_clouds)
        (folder / "rollout.partial.mp4").replace(folder / "rollout.mp4")
        result = {
            "environment": self.cli.environment,
            "environment_name": self.environment["name"],
            "episode": episode,
            "condition": condition,
            "layout_seed": record["layout_seed"],
            "policy_seed": policy_seed,
            "initial_state_hash": record["state_hash"],
            "success": bool(success),
            "safe_success": bool(success and not collision_any),
            "collision_any": bool(collision_any),
            "success_with_collision": bool(success and collision_any),
            "collision_failure": bool(not success and collision_any),
            "ncr": bool(not success and not collision_any),
            "perception_failure": bool(perception_failure),
            "perception_failure_step": perception_failure_step,
            "steps": step,
            "chunks": len(chunk_records),
            "first_collision_step": first_collision_step,
            "first_strict_success_step": first_strict_success_step,
            "action_finite": True,
            "action_contract": "absolute world EEF pose (xyz, axis-angle, gripper), OSC_POSE",
            "absolute_eef_contract_valid": action_contract_violation_count == 0,
            "action_contract_violation_count": action_contract_violation_count,
            "environment_action_spec_exceedance_count": action_spec_exceedance_count,
            "environment_action_spec_exceedance_max": action_spec_exceedance_max,
            "termination": (
                "perception_failure"
                if perception_failure
                else "success"
                if success
                else "env_done"
                if done
                else "horizon"
            ),
            "wall_seconds": time.monotonic() - started,
            "video_frames": len(video_steps),
            "folder": str(folder.relative_to(self.cli.output)),
        }
        dump_json(folder / "result.json", result)
        print(json.dumps(json_safe(result)), flush=True)
        return result

    def close(self) -> None:
        self.env.close()


def compare_pair(folder: Path) -> dict[str, Any]:
    source_folder = folder / CONDITIONS[0]
    perceived_folder = folder / CONDITIONS[1]
    source = json.loads((source_folder / "result.json").read_text())
    perceived = json.loads((perceived_folder / "result.json").read_text())
    source_initial = json.loads((source_folder / "initial_audit.json").read_text())
    perceived_initial = json.loads((perceived_folder / "initial_audit.json").read_text())
    if source["initial_state_hash"] != perceived["initial_state_hash"]:
        raise RuntimeError("paired initial state hashes differ")
    if source["policy_seed"] != perceived["policy_seed"]:
        raise RuntimeError("paired policy seeds differ")
    for key in ("rgb_sha256", "depth_sha256", "calibration_hashes"):
        if source_initial[key] != perceived_initial[key]:
            raise RuntimeError(f"paired initial {key} differs")
    if source_initial["rng_after_perception"] != perceived_initial["rng_after_perception"]:
        raise RuntimeError("paired policy RNG differs after perception initialization")
    common_obs = set(source_initial["native_observation_hashes"]) & set(
        perceived_initial["native_observation_hashes"]
    )
    common_obs.discard("task_pointcloud")
    for key in common_obs:
        if (
            source_initial["native_observation_hashes"][key]
            != perceived_initial["native_observation_hashes"][key]
        ):
            raise RuntimeError(f"paired initial native observation differs: {key}")
    result = {
        "episode": source["episode"],
        "initial_state_equal": True,
        "initial_rgbd_equal": True,
        "calibration_equal": True,
        "policy_seed_equal": True,
        "initial_policy_rng_equal": True,
        "native_non_pointcloud_observations_equal": True,
        "oracle_provider_enabled_only_for_source": bool(
            source_initial["oracle_provider_enabled"]
            and not perceived_initial["oracle_provider_enabled"]
        ),
        "task_rescue": bool(not source["success"] and perceived["success"]),
        "task_regression": bool(source["success"] and not perceived["success"]),
        "safe_rescue": bool(
            not source["safe_success"] and perceived["safe_success"]
        ),
        "safe_regression": bool(
            source["safe_success"] and not perceived["safe_success"]
        ),
    }
    dump_json(folder / "pair_audit.json", result)
    return result


def source_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in Path(__file__).parent.glob("*.py")
        if not path.name.startswith("test_")
    }


def git_heads() -> dict[str, str]:
    return {
        name: subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        for name, path in (
            ("root", ROOT),
            ("robomimic", ROOT / "third_party/robomimic"),
            ("robosuite", ROOT / "third_party/robosuite"),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--stage", choices=("pilot", "expansion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--state-bank", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.weights = args.weights.resolve()
    args.locked_config = args.locked_config.resolve()
    environment = ENVIRONMENTS[args.environment]
    args.state_bank = (
        Path(environment["state_bank"]) if args.state_bank is None else args.state_bank
    ).resolve()
    if args.horizon != 600:
        raise ValueError("formal comparison locks H600")
    episodes = (
        PILOT_EPISODES
        if args.stage == "pilot"
        else tuple(index for index in range(50) if index not in PILOT_EPISODES)
    )
    stage_dir = args.output / "rollouts" / args.environment / args.stage
    if stage_dir.exists():
        raise FileExistsError(stage_dir)
    stage_dir.mkdir(parents=True)

    weights_manifest = verify_manifest(args.weights / "manifest.json")
    force_offline_runtime()
    locked = json.loads(args.locked_config.read_text())
    if locked.get("weights_manifest_sha256") != sha256_file(
        args.weights / "manifest.json"
    ):
        raise RuntimeError("locked prompt config does not match the weight manifest")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("formal comparison requires CUDA")
    manifest = {
        "format_version": 1,
        "environment": args.environment,
        "environment_name": environment["name"],
        "bank_seed": environment["seed"],
        "stage": args.stage,
        "episodes": list(episodes),
        "conditions": list(CONDITIONS),
        "pair_order": list(CONDITIONS),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "state_bank": str(args.state_bank),
        "state_bank_sha256": sha256_file(args.state_bank),
        "locked_config": str(args.locked_config),
        "locked_config_sha256": sha256_file(args.locked_config),
        "weights_manifest": weights_manifest,
        "horizon": args.horizon,
        "success_criterion": "old strict OR released in tray continuously for 0.5 seconds",
        "perceived_runtime_inputs": [
            "agentview uint8 RGB",
            "agentview metric depth",
            "fixed camera intrinsics/extrinsics",
            "locked manual role prompts",
        ],
        "oracle_audit_only": [
            "MuJoCo geom segmentation",
            "object poses",
            "contacts",
            "grasp/release state",
        ],
        "source_hashes": source_hashes(),
        "git_heads": git_heads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    dump_json(stage_dir / "manifest.json", manifest)

    evaluation = Evaluation(args)
    completed = []
    try:
        for episode in episodes:
            episode_folder = stage_dir / f"ep{episode:02d}"
            pair_results = {}
            for condition in CONDITIONS:
                pair_results[condition] = evaluation.run_condition(
                    episode, condition, episode_folder / condition
                )
            pair_audit = compare_pair(episode_folder)
            completed.append(
                {
                    "episode": episode,
                    "results": pair_results,
                    "pair_audit": pair_audit,
                }
            )
            dump_json(stage_dir / "progress.json", completed)
    finally:
        evaluation.close()
    dump_json(
        stage_dir / "complete.json",
        {"episodes": list(episodes), "pairs": len(completed), "rollouts": 2 * len(completed)},
    )


if __name__ == "__main__":
    main()
