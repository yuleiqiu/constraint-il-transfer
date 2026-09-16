"""Shared state-bank and corrected-reset utilities for this experiment."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/robomimic/train/lan_o3dp_tabletop_can_wide_support_abs_pose_sample_100demo_seed42"
    / "20260826184200/models"
    / "model_epoch_50_tabletop_can_wide_support_pointcloud_abs_eef_100_success_0.8.pth"
)
ENVIRONMENTS = {
    "clean": {
        "name": "PickPlaceCanTabletop",
        "seed": 827,
        "state_bank": ROOT
        / "outputs/tabletop_wide_support/fixed_clean_state_bank_seed827_n50/initial_state_bank.pkl",
        "has_obstacle": False,
    },
    "raised_milk": {
        "name": "PickPlaceCanTabletopRaisedMilk",
        "seed": 828,
        "state_bank": ROOT
        / "outputs/tabletop_wide_support/raised_milk_state_bank_seed828_n50/initial_state_bank.pkl",
        "has_obstacle": True,
    },
}
PILOT_EPISODES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
CONDITIONS = ("oracle_mask_source", "perceived_mask_source")


def audit_absolute_eef_actions(actions: np.ndarray) -> dict[str, Any]:
    """Validate the project's executable absolute-world OSC_POSE contract.

    ``env.action_spec`` is the generic normalized robosuite interface and is
    not an appropriate bound for this dataset: XYZ is expressed in world
    metres and orientation is a canonical axis-angle vector.  The executable
    contract therefore requires seven finite values, a rotation-vector norm no
    greater than pi, and a normalized gripper command.  World XYZ has no
    component-wise normalized bound.
    """

    values = np.asarray(actions)
    shape_valid = values.ndim == 2 and values.shape[1:] == (7,)
    action_count = int(values.shape[0]) if values.ndim >= 1 else 0
    if not shape_valid:
        return {
            "valid": False,
            "shape_valid": False,
            "finite": False,
            "action_count": action_count,
            "violation_count": action_count or 1,
            "max_axis_angle_norm_rad": None,
            "max_gripper_abs": None,
            "contract": (
                "7-D finite absolute world XYZ + canonical axis-angle "
                "(norm <= pi) + gripper in [-1.0001, 1.0001]"
            ),
        }

    finite_rows = np.isfinite(values).all(axis=1)
    safe = np.where(np.isfinite(values), values, 0.0)
    rotation_norm = np.linalg.norm(safe[:, 3:6], axis=1)
    gripper_abs = np.abs(safe[:, 6])
    rotation_valid = rotation_norm <= np.pi + 1e-4
    gripper_valid = gripper_abs <= 1.0001
    valid_rows = finite_rows & rotation_valid & gripper_valid
    return {
        "valid": bool(np.all(valid_rows)),
        "shape_valid": True,
        "finite": bool(np.all(finite_rows)),
        "action_count": action_count,
        "violation_count": int(np.count_nonzero(~valid_rows)),
        "max_axis_angle_norm_rad": (
            float(np.max(rotation_norm)) if action_count else 0.0
        ),
        "max_gripper_abs": float(np.max(gripper_abs)) if action_count else 0.0,
        "contract": (
            "7-D finite absolute world XYZ + canonical axis-angle "
            "(norm <= pi) + gripper in [-1.0001, 1.0001]"
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(state["states"]).tobytes())
    digest.update((state.get("model") or "").encode("utf-8"))
    digest.update((state.get("ep_meta") or "").encode("utf-8"))
    return digest.hexdigest()


def load_state_bank(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        bank = pickle.load(stream)
    if bank.get("format_version") != 1 or not isinstance(bank.get("states"), dict):
        raise ValueError(f"unsupported state bank: {path}")
    return bank


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class CorrectedResetEnv:
    """Synchronize saved physical bodies with stale Python reset metadata."""

    def __init__(self, env) -> None:
        from robomimic.envs.wrappers import EnvWrapper

        self.env = env
        current = env
        while isinstance(current, EnvWrapper):
            current = current.env
        self.base_env = current
        self.raw = current.env
        self.layout_seed = None
        self.reset_audit: dict[str, Any] = {}
        self.initial_obs: dict[str, np.ndarray] = {}
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_done = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset_to(self, state):
        if self.layout_seed is None:
            raise RuntimeError("layout_seed must be set before reset_to")
        self.raw.rng = np.random.default_rng(self.layout_seed)
        obs = self.env.reset_to(state)
        goal_before = None if self.raw.goal_xy is None else self.raw.goal_xy.copy()
        obstacle_before = (
            None if self.raw.obstacle_xy is None else self.raw.obstacle_xy.copy()
        )
        physical_goal = self.raw.sim.model.body_pos[
            self.raw.goal_tray_body_id, :2
        ].copy()
        physical_obstacle = None
        if self.raw.obstacle_body_id is not None:
            physical_obstacle = self.raw.sim.model.body_pos[
                self.raw.obstacle_body_id, :2
            ].copy()
        self.raw.goal_xy = physical_goal.copy()
        self.raw.obstacle_xy = (
            None if physical_obstacle is None else physical_obstacle.copy()
        )
        restored_hash = state_hash(self.env.get_state())
        self.reset_audit = {
            "goal_before": goal_before,
            "obstacle_before": obstacle_before,
            "physical_goal": physical_goal,
            "physical_obstacle": physical_obstacle,
            "restored_state_hash": restored_hash,
        }
        self.initial_obs = {key: np.asarray(value).copy() for key, value in obs.items()}
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_done = False
        return obs

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        result = self.env.step(action)
        self.last_done = bool(result[2])
        return result

    def close(self) -> None:
        close = getattr(self.raw, "close", None)
        if callable(close):
            close()
