from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EnvironmentConfig, PolicyConfig
from .env import LayoutSpec, PointNavEnv, PointNavState, make_layout
from .planner import oracle_actions


SCHEMA_VERSION = "toy-recovery-trajectory-v1"
EVALUATION_SCHEMA_VERSION = "toy-recovery-evaluation-v1"


@dataclass
class Trajectory:
    rgb: np.ndarray
    state: np.ndarray
    action: np.ndarray
    collision: np.ndarray
    success: np.ndarray
    layout: LayoutSpec
    policy_seed: int = 0
    source: str = "oracle"

    def validate(self) -> None:
        steps = self.action.shape[0]
        if self.rgb.shape != (steps + 1, 64, 64, 3):
            raise ValueError(f"invalid RGB shape {self.rgb.shape}")
        if self.state.shape != (steps + 1, 13):
            raise ValueError(f"invalid state shape {self.state.shape}")
        if self.action.shape[1:] != (2,):
            raise ValueError(f"invalid action shape {self.action.shape}")
        if self.collision.shape != (steps,) or self.success.shape != (steps,):
            raise ValueError("terminal arrays must align with actions")
        if not np.isfinite(self.action).all():
            raise ValueError("actions must be finite")


@dataclass(frozen=True)
class EvaluationRecord:
    method: str
    difficulty: int
    layout_seed: int
    layout_id: str
    policy_seed: int
    success: bool
    collision: bool
    timeout: bool
    steps: int
    path_length: float
    minimum_clearance: float | None
    guidance_activations: int
    inference_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EVALUATION_SCHEMA_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "EvaluationRecord":
        values = dict(raw)
        schema = values.pop("schema", EVALUATION_SCHEMA_VERSION)
        if schema != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema {schema}")
        return cls(**values)


def normalize_actions(actions: np.ndarray | torch.Tensor, max_step: float = 0.05):
    if torch.is_tensor(actions):
        return torch.clamp(actions / max_step, -1.0, 1.0)
    return np.clip(np.asarray(actions) / max_step, -1.0, 1.0)


def denormalize_actions(actions: np.ndarray | torch.Tensor, max_step: float = 0.05):
    if torch.is_tensor(actions):
        return torch.clamp(actions, -1.0, 1.0) * max_step
    return np.clip(np.asarray(actions), -1.0, 1.0) * max_step


def collect_trajectory(
    layout: LayoutSpec,
    actions: np.ndarray,
    config: EnvironmentConfig | None = None,
    *,
    policy_seed: int = 0,
    source: str = "oracle",
) -> Trajectory:
    cfg = config or EnvironmentConfig()
    env = PointNavEnv(layout=layout, config=cfg)
    obs = env.reset(layout.seed, layout)
    rgbs, states = [obs["rgb"]], [obs["state"]]
    executed, collisions, successes = [], [], []
    done = False
    for action in actions:
        obs, _, done, info = env.step(action)
        executed.append(info["executed_action"])
        collisions.append(info["collision"])
        successes.append(info["success"])
        rgbs.append(obs["rgb"])
        states.append(obs["state"])
        if done:
            break
    trajectory = Trajectory(
        rgb=np.asarray(rgbs, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(executed, dtype=np.float32).reshape(-1, 2),
        collision=np.asarray(collisions, dtype=np.bool_),
        success=np.asarray(successes, dtype=np.bool_),
        layout=layout,
        policy_seed=policy_seed,
        source=source,
    )
    trajectory.validate()
    return trajectory


def generate_oracle_demonstrations(
    count: int,
    *,
    difficulty: int = 0,
    seed_start: int = 0,
    config: EnvironmentConfig | None = None,
) -> list[Trajectory]:
    cfg = config or EnvironmentConfig()
    trajectories = []
    for offset in range(count):
        seed = seed_start + offset
        layout = make_layout(difficulty, seed, cfg)
        mode = "shortest"
        if difficulty == 1:
            mode = "upper" if offset % 2 == 0 else "lower"
        actions = oracle_actions(layout, cfg, mode=mode)
        trajectory = collect_trajectory(layout, actions, cfg)
        if not trajectory.success[-1] or trajectory.collision.any():
            raise RuntimeError(f"oracle failed generated layout {layout.identifier}")
        trajectories.append(trajectory)
    return trajectories


def write_trajectories(
    path: str | Path,
    trajectories: Iterable[Trajectory],
    *,
    append: bool = False,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with h5py.File(target, mode) as handle:
        if "schema" in handle.attrs and handle.attrs["schema"] != SCHEMA_VERSION:
            raise ValueError("cannot append to a different schema")
        handle.attrs["schema"] = SCHEMA_VERSION
        episodes = handle.require_group("episodes")
        start = len(episodes)
        for index, trajectory in enumerate(trajectories, start=start):
            trajectory.validate()
            group = episodes.create_group(f"episode_{index:06d}")
            group.create_dataset(
                "rgb",
                data=trajectory.rgb,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=1,
            )
            group.create_dataset("state", data=trajectory.state, dtype=np.float32)
            group.create_dataset("action", data=trajectory.action, dtype=np.float32)
            group.create_dataset(
                "collision", data=trajectory.collision, dtype=np.bool_
            )
            group.create_dataset("success", data=trajectory.success, dtype=np.bool_)
            group.attrs["layout"] = json.dumps(
                trajectory.layout.to_dict(), sort_keys=True
            )
            group.attrs["policy_seed"] = trajectory.policy_seed
            group.attrs["source"] = trajectory.source
        handle.attrs["num_episodes"] = len(episodes)


def iter_trajectories(path: str | Path) -> Iterator[Trajectory]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory schema")
        for name in sorted(handle["episodes"]):
            group = handle["episodes"][name]
            trajectory = Trajectory(
                rgb=group["rgb"][...],
                state=group["state"][...],
                action=group["action"][...],
                collision=group["collision"][...].astype(bool),
                success=group["success"][...].astype(bool),
                layout=LayoutSpec.from_dict(json.loads(group.attrs["layout"])),
                policy_seed=int(group.attrs["policy_seed"]),
                source=str(group.attrs["source"]),
            )
            trajectory.validate()
            yield trajectory


def replay_trajectory(
    trajectory: Trajectory,
    config: EnvironmentConfig | None = None,
    *,
    atol: float = 1e-6,
) -> bool:
    cfg = config or EnvironmentConfig()
    env = PointNavEnv(layout=trajectory.layout, config=cfg)
    obs = env.reset(trajectory.layout.seed, trajectory.layout)
    if not np.array_equal(obs["rgb"], trajectory.rgb[0]):
        return False
    if not np.allclose(obs["state"], trajectory.state[0], atol=atol):
        return False
    for index, action in enumerate(trajectory.action):
        obs, _, _, info = env.step(action)
        if not np.array_equal(obs["rgb"], trajectory.rgb[index + 1]):
            return False
        if not np.allclose(obs["state"], trajectory.state[index + 1], atol=atol):
            return False
        if bool(info["collision"]) != bool(trajectory.collision[index]):
            return False
        if bool(info["success"]) != bool(trajectory.success[index]):
            return False
    return True


def write_evaluation_records(
    path: str | Path,
    records: Iterable[EvaluationRecord],
    *,
    append: bool = False,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_evaluation_records(path: str | Path) -> list[EvaluationRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(EvaluationRecord.from_dict(json.loads(line)))
    return records


class TrajectoryWindowDataset(Dataset):
    """Diffusion-policy windows with fixed 2/16 observation/action horizons."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        policy_config: PolicyConfig | None = None,
        *,
        max_step: float = 0.05,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.policy_config = policy_config or PolicyConfig()
        self.max_step = max_step
        self.index: list[tuple[int, str, int]] = []
        self.difficulties: set[int] = set()
        for path_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as handle:
                for episode_name in sorted(handle["episodes"]):
                    episode = handle["episodes"][episode_name]
                    layout = json.loads(episode.attrs["layout"])
                    self.difficulties.add(int(layout["difficulty"]))
                    action_count = episode["action"].shape[0]
                    for step in range(action_count):
                        self.index.append((path_index, episode_name, step))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path_index, episode_name, step = self.index[index]
        p = self.policy_config
        with h5py.File(self.paths[path_index], "r") as handle:
            episode = handle["episodes"][episode_name]
            first_obs = max(0, step - p.observation_horizon + 1)
            history = episode["rgb"][first_obs : step + 1]
            if history.shape[0] < p.observation_horizon:
                padding = np.repeat(
                    history[:1],
                    p.observation_horizon - history.shape[0],
                    axis=0,
                )
                history = np.concatenate((padding, history), axis=0)
            # Match robomimic SequenceDataset with frame_stack=2,
            # seq_length=16, and pad_same=True. The predicted sequence starts
            # one action before the current observation, and inference executes
            # [observation_horizon - 1 : + action_horizon]. Boundary values are
            # repeated instead of introducing artificial zero-action targets.
            sequence_indices = (
                np.arange(p.prediction_horizon)
                + step
                - (p.observation_horizon - 1)
            )
            action_count = episode["action"].shape[0]
            valid = (sequence_indices >= 0) & (sequence_indices < action_count)
            clipped_indices = np.clip(sequence_indices, 0, action_count - 1)
            actions = episode["action"][...][clipped_indices]
        return {
            "rgb": torch.from_numpy(history.copy()).permute(0, 3, 1, 2).float()
            / 255.0,
            "action": torch.from_numpy(
                normalize_actions(actions, self.max_step).astype(np.float32)
            ),
            "valid": torch.from_numpy(valid.copy()),
        }
