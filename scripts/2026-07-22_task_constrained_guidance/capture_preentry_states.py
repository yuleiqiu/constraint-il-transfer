"""Capture target-directed obstructed states before the EEF enters a footprint."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GUIDED_DIR = SCRIPT_DIR.parent / "guided_denoising"
for directory in (SCRIPT_DIR, GUIDED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common as guided_common  # noqa: E402
from feasibility import (  # noqa: E402
    closest_target_progress,
    minimum_segment_clearance_xy,
    reconstruct_positions,
)
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def point_clearance_xy(position, centers, radii):
    if len(radii) == 0:
        return float("inf")
    distances = np.linalg.norm(
        np.asarray(centers)[:, :2] - np.asarray(position)[None, :2], axis=1
    )
    return float(np.min(distances - np.asarray(radii)))


def write_state(states_group, state_id, payload):
    item = states_group.create_group(state_id)
    for key in (
        "env",
        "seed",
        "episode",
        "chunk",
        "start_step",
        "noise_seed",
        "predicted_progress_m",
        "predicted_min_clearance_m",
        "start_min_clearance_m",
        "collision_lead_chunks",
        "collision_lead_steps",
    ):
        item.attrs[key] = payload[key]
    item.create_dataset("sim_state", data=payload["state"]["states"])
    item.create_dataset("model_xml", data=payload["state"]["model"], dtype=STRING_DTYPE)
    item.create_dataset(
        "ep_meta", data=payload["state"].get("ep_meta", ""), dtype=STRING_DTYPE
    )
    obs_group = item.create_group("obs")
    for key, value in payload["obs"].items():
        kwargs = {"compression": "gzip"} if np.asarray(value).ndim >= 3 else {}
        obs_group.create_dataset(key, data=value, **kwargs)
    item.create_dataset("obstacle_centers", data=payload["centers"])
    item.create_dataset("obstacle_radii", data=payload["radii"])
    item.create_dataset(
        "obstacle_names",
        data=np.asarray(payload["obstacle_names"], dtype=object),
        dtype=STRING_DTYPE,
    )
    item.create_dataset("target_position", data=payload["target_position"])
    item.create_dataset("baseline_actions", data=payload["baseline_actions"])
    item.create_dataset("predicted_positions", data=payload["predicted_positions"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=guided_common.DEFAULT_AGENT)
    parser.add_argument("--envs", nargs="+", default=["PickPlaceBreadCerealCan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[600, 601, 602])
    parser.add_argument("--states-per-env", type=int, default=5)
    parser.add_argument("--max-episodes-per-seed", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--minimum-start-clearance", type=float, default=0.0)
    parser.add_argument("--predicted-clearance-threshold", type=float, default=0.0)
    parser.add_argument("--minimum-collision-lead-chunks", type=int, default=0)
    parser.add_argument("--max-collision-lead-chunks", type=int, default=1)
    parser.add_argument("--target-object-name", default="Can")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/task_constrained_guidance/preentry_states"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.minimum_collision_lead_chunks < 0:
        raise ValueError("--minimum-collision-lead-chunks must be nonnegative")
    if args.max_collision_lead_chunks < args.minimum_collision_lead_chunks:
        raise ValueError(
            "--max-collision-lead-chunks must be at least the minimum"
        )
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError("{} exists; pass --overwrite".format(args.out_dir))
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = guided_common.load_guided_policy_and_checkpoint(args.agent, device)
    counts = {env_name: 0 for env_name in args.envs}
    episodes_attempted = {env_name: 0 for env_name in args.envs}
    states_path = args.out_dir / "states.hdf5"
    with h5py.File(states_path, "w") as output:
        states_group = output.create_group("states")
        for env_name in args.envs:
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=ckpt,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                verbose=False,
            )
            for seed in args.seeds:
                if counts[env_name] >= args.states_per_env:
                    break
                for episode in range(args.max_episodes_per_seed):
                    if counts[env_name] >= args.states_per_env:
                        break
                    episodes_attempted[env_name] += 1
                    if episodes_attempted[env_name] % 10 == 0:
                        print(
                            "searched {} episodes for {}; captured {}".format(
                                episodes_attempted[env_name],
                                env_name,
                                counts[env_name],
                            ),
                            flush=True,
                        )
                    guided_common.seed_everything(seed * 1000 + episode)
                    policy.start_episode()
                    obs = env.reset()
                    raw_env = guided_common.raw_env_from_wrapper(env)
                    records = guided_common.active_object_records(raw_env)
                    maps = guided_common.contact_maps(
                        raw_env, records, args.target_object_name
                    )
                    step = 0
                    chunk_index = 0
                    target_seen = False
                    done = False
                    candidates = []
                    while step < args.horizon and not done and not target_seen:
                        start_step = step
                        state_before = env.get_state()
                        obs_before = guided_common.policy_observation(obs, policy)
                        centers, radii, obstacle_names = (
                            guided_common.oracle_obstacle_geometry(
                                raw_env, records, args.target_object_name
                            )
                        )
                        target_position = guided_common.target_object_position(
                            raw_env, records, args.target_object_name
                        )
                        start_position = guided_common.current_eef_position(obs_before)
                        noise_seed = seed * 1_000_000 + episode * 1000 + chunk_index
                        context = guided_common.make_guidance_context(
                            policy,
                            obs_before,
                            centers,
                            radii,
                            scale=0.0,
                            clearance_margin=0.0,
                        )
                        _, baseline_actions, _ = guided_common.sample_action_chunk(
                            policy, obs_before, context, noise_seed=noise_seed
                        )
                        predicted_positions = reconstruct_positions(
                            start_position, baseline_actions
                        )
                        progress = closest_target_progress(
                            start_position, predicted_positions, target_position
                        )
                        clearance = minimum_segment_clearance_xy(
                            start_position, predicted_positions, centers, radii
                        )
                        start_clearance = point_clearance_xy(
                            start_position, centers, radii
                        )
                        predicted_candidate = (
                            progress > 0.0
                            and clearance <= args.predicted_clearance_threshold
                            and start_clearance >= args.minimum_start_clearance
                        )

                        _, obstacle_contacts_at_start, _ = (
                            guided_common.contact_snapshot(raw_env, maps)
                        )
                        if predicted_candidate and obstacle_contacts_at_start == 0:
                            candidates.append({
                                "env": env_name,
                                "seed": seed,
                                "episode": episode,
                                "chunk": chunk_index,
                                "start_step": start_step,
                                "noise_seed": noise_seed,
                                "predicted_progress_m": progress,
                                "predicted_min_clearance_m": clearance,
                                "start_min_clearance_m": start_clearance,
                                "state": state_before,
                                "obs": obs_before,
                                "centers": centers,
                                "radii": radii,
                                "obstacle_names": obstacle_names,
                                "target_position": target_position,
                                "baseline_actions": baseline_actions,
                                "predicted_positions": predicted_positions,
                            })

                        collision_in_prefix = False
                        target_before_prefix = target_seen
                        for action in baseline_actions:
                            if step >= args.horizon:
                                break
                            obs, _, done, _ = env.step(action)
                            target_contacts, obstacle_contacts, _ = (
                                guided_common.contact_snapshot(raw_env, maps)
                            )
                            target_seen = target_seen or target_contacts > 0
                            collision_in_prefix = collision_in_prefix or obstacle_contacts > 0
                            step += 1
                            if done or guided_common.task_success(env):
                                done = True
                                break

                        if (
                            collision_in_prefix
                            and not target_before_prefix
                            and not target_seen
                            and candidates
                        ):
                            valid_candidates = [
                                candidate
                                for candidate in candidates
                                if args.minimum_collision_lead_chunks
                                <= chunk_index - candidate["chunk"]
                                <= args.max_collision_lead_chunks
                            ]
                            if not valid_candidates:
                                break
                            latest_candidate = valid_candidates[-1]
                            collision_lead_chunks = chunk_index - latest_candidate["chunk"]
                            latest_candidate["collision_lead_chunks"] = collision_lead_chunks
                            latest_candidate["collision_lead_steps"] = (
                                start_step - latest_candidate["start_step"]
                            )
                            state_id = "{}_preentry_{:03d}".format(
                                env_name, counts[env_name]
                            )
                            write_state(
                                states_group,
                                state_id,
                                latest_candidate,
                            )
                            counts[env_name] += 1
                            print(
                                "captured {} progress={:.4f} clearance={:.4f} "
                                "start_clearance={:.4f} lead_chunks={}".format(
                                    state_id,
                                    latest_candidate["predicted_progress_m"],
                                    latest_candidate["predicted_min_clearance_m"],
                                    latest_candidate["start_min_clearance_m"],
                                    latest_candidate["collision_lead_chunks"],
                                ),
                                flush=True,
                            )
                            break
                        # Once the first pre-target obstacle contact has happened,
                        # later chunks can no longer be pre-entry candidates for
                        # this layout. Move to the next episode immediately.
                        if collision_in_prefix and not target_before_prefix:
                            break
                        chunk_index += 1

    manifest = {
        "agent": str(args.agent),
        "envs": args.envs,
        "seeds": args.seeds,
        "states_per_env": args.states_per_env,
        "max_episodes_per_seed": args.max_episodes_per_seed,
        "horizon": args.horizon,
        "minimum_start_clearance_m": args.minimum_start_clearance,
        "predicted_clearance_threshold_m": args.predicted_clearance_threshold,
        "minimum_collision_lead_chunks": args.minimum_collision_lead_chunks,
        "max_collision_lead_chunks": args.max_collision_lead_chunks,
        "selection": (
            "latest pre-contact chunk with closest 3-D progress > 0, predicted swept "
            "XY clearance <= threshold, chunk-start XY point clearance >= minimum, "
            "and no contact at its start; selected when the layout later has its first "
            "pre-target obstacle contact"
        ),
        "frozen_horizons": {"observation": 2, "prediction": 16, "action": 8},
        "captured_states": counts,
        "episodes_attempted": episodes_attempted,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
