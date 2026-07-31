from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import EnvironmentConfig


@dataclass(frozen=True, order=True)
class Obstacle:
    x: float
    y: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray((self.x, self.y), dtype=np.float64)


@dataclass(frozen=True)
class LayoutSpec:
    difficulty: int
    seed: int
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacles: tuple[Obstacle, ...] = ()
    template: str = "pointnav-v1"

    def __post_init__(self) -> None:
        if self.difficulty not in (0, 1, 2):
            raise ValueError("difficulty must be D0, D1, or D2")
        if len(self.obstacles) != self.difficulty:
            raise ValueError("obstacle count must match difficulty")

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "seed": self.seed,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [asdict(o) for o in self.obstacles],
            "template": self.template,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LayoutSpec":
        return cls(
            difficulty=int(raw["difficulty"]),
            seed=int(raw["seed"]),
            start=tuple(float(v) for v in raw["start"]),
            goal=tuple(float(v) for v in raw["goal"]),
            obstacles=tuple(
                Obstacle(**{k: float(v) for k, v in obstacle.items()})
                for obstacle in raw["obstacles"]
            ),
            template=str(raw.get("template", "pointnav-v1")),
        )

    @property
    def identifier(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class PointNavState:
    position: np.ndarray
    step_count: int
    collision: bool
    success: bool
    path_length: float
    minimum_clearance: float
    layout: LayoutSpec

    @property
    def done(self) -> bool:
        return self.collision or self.success or self.step_count >= 80

    def copy(self) -> "PointNavState":
        return PointNavState(
            position=self.position.copy(),
            step_count=self.step_count,
            collision=self.collision,
            success=self.success,
            path_length=self.path_length,
            minimum_clearance=self.minimum_clearance,
            layout=self.layout,
        )

    def vector(self, max_obstacles: int = 2) -> np.ndarray:
        values = [
            *self.position.tolist(),
            *self.layout.goal,
            float(self.step_count),
            float(self.collision),
            float(self.success),
        ]
        for index in range(max_obstacles):
            if index < len(self.layout.obstacles):
                obstacle = self.layout.obstacles[index]
                values.extend((obstacle.x, obstacle.y, obstacle.radius))
            else:
                values.extend((0.0, 0.0, 0.0))
        return np.asarray(values, dtype=np.float32)


def _segment_point_distance(
    start: np.ndarray, end: np.ndarray, point: np.ndarray
) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, delta) / denominator)
    projection = start + np.clip(fraction, 0.0, 1.0) * delta
    return float(np.linalg.norm(point - projection))


def swept_clearance(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Iterable[Obstacle],
    agent_radius: float,
) -> float:
    clearances = [
        _segment_point_distance(start, end, obstacle.center)
        - obstacle.radius
        - agent_radius
        for obstacle in obstacles
    ]
    return min(clearances, default=math.inf)


def _raw_layout(
    difficulty: int,
    seed: int,
    attempt: int,
    config: EnvironmentConfig,
) -> LayoutSpec:
    rng = np.random.default_rng(np.random.SeedSequence([seed, difficulty, attempt]))
    start = (-0.78 + rng.uniform(-0.07, 0.07), rng.uniform(-0.28, 0.28))
    goal = (0.78 + rng.uniform(-0.07, 0.07), rng.uniform(-0.28, 0.28))

    def line_y_at(x: float) -> float:
        fraction = (x - start[0]) / (goal[0] - start[0])
        return float(start[1] + fraction * (goal[1] - start[1]))

    if difficulty == 0:
        obstacles: tuple[Obstacle, ...] = ()
        template = "pointnav-v1"
    elif difficulty == 1:
        x = rng.uniform(-0.08, 0.08)
        radius = 0.07
        side = 1.0 if seed % 2 == 0 else -1.0
        tangent_fraction = 0.85
        obstacles = (
            Obstacle(
                x=x,
                y=line_y_at(x)
                + side
                * tangent_fraction
                * (radius + config.agent_radius),
                radius=radius,
            ),
        )
        template = "pointnav-v2-tangent-d1"
    else:
        # Two separated near-tangent blockers on the same side of the direct
        # route. Passing requires maintaining a lateral detour across multiple
        # action chunks, while leaving a small natural source success set.
        side = 1.0 if seed % 2 == 0 else -1.0
        generated = []
        for index, nominal_x in enumerate((-0.22, 0.22)):
            x = nominal_x + rng.uniform(-0.035, 0.035)
            radius = 0.07
            tangent_fraction = 0.78 + 0.04 * index
            generated.append(
                Obstacle(
                    x=x,
                    y=line_y_at(x)
                    + side
                    * tangent_fraction
                    * (radius + config.agent_radius),
                    radius=radius,
                )
            )
        obstacles = tuple(generated)
        template = "pointnav-v2-tangent-d2"
    return LayoutSpec(
        difficulty=difficulty,
        seed=seed,
        start=start,
        goal=goal,
        obstacles=tuple(sorted(obstacles)),
        template=template,
    )


def make_layout(
    difficulty: int,
    seed: int,
    config: EnvironmentConfig | None = None,
    max_attempts: int = 128,
) -> LayoutSpec:
    """Create a deterministic, template-jittered layout with a safe path."""

    from .planner import d1_mode_path, path_to_actions, shortest_path

    cfg = config or EnvironmentConfig()
    for attempt in range(max_attempts):
        layout = _raw_layout(difficulty, seed, attempt, cfg)
        path = shortest_path(layout, cfg, margin=cfg.planner_margin)
        direct_clearance = swept_clearance(
            np.asarray(layout.start),
            np.asarray(layout.goal),
            layout.obstacles,
            cfg.agent_radius,
        )
        direct_block_count = sum(
            _segment_point_distance(
                np.asarray(layout.start),
                np.asarray(layout.goal),
                obstacle.center,
            )
            <= obstacle.radius + cfg.agent_radius
            for obstacle in layout.obstacles
        )
        blocks_direct = difficulty == 0 or direct_clearance <= 0.0
        structure_ok = True
        if difficulty == 1:
            structure_ok = all(
                d1_mode_path(layout, cfg, mode=mode) is not None
                for mode in ("upper", "lower")
            )
        elif difficulty == 2 and path is not None:
            actions = path_to_actions(path, cfg.max_step, ease=False)
            positions = np.concatenate(
                (
                    np.asarray(layout.start, dtype=np.float64)[None],
                    np.asarray(layout.start, dtype=np.float64)[None]
                    + np.cumsum(actions, axis=0),
                ),
                axis=0,
            )
            nearest_indices = [
                int(np.argmin(np.linalg.norm(positions - obstacle.center, axis=1)))
                for obstacle in layout.obstacles
            ]
            structure_ok = (
                direct_block_count == 2
                and max(nearest_indices) - min(nearest_indices) >= 8
            )
        if path is not None and blocks_direct and structure_ok:
            return layout
    raise RuntimeError(f"could not create feasible D{difficulty} layout seed {seed}")


class PointNavEnv:
    def __init__(
        self,
        difficulty: int = 0,
        config: EnvironmentConfig | None = None,
        layout: LayoutSpec | None = None,
    ) -> None:
        self.difficulty = difficulty
        self.config = config or EnvironmentConfig()
        self.layout = layout
        self.state: PointNavState | None = None

    def reset(
        self, seed: int, layout: LayoutSpec | None = None
    ) -> dict[str, np.ndarray]:
        self.layout = layout or make_layout(self.difficulty, seed, self.config)
        self.difficulty = self.layout.difficulty
        self.state = PointNavState(
            position=np.asarray(self.layout.start, dtype=np.float64),
            step_count=0,
            collision=False,
            success=False,
            path_length=0.0,
            minimum_clearance=math.inf,
            layout=self.layout,
        )
        self._update_point_clearance()
        return self.observation()

    def _require_state(self) -> PointNavState:
        if self.state is None:
            raise RuntimeError("reset must be called before using the environment")
        return self.state

    def _update_point_clearance(self) -> None:
        state = self._require_state()
        clearance = min(
            (
                float(np.linalg.norm(state.position - obstacle.center))
                - obstacle.radius
                - self.config.agent_radius
                for obstacle in state.layout.obstacles
            ),
            default=math.inf,
        )
        state.minimum_clearance = min(state.minimum_clearance, clearance)

    def observation(self) -> dict[str, np.ndarray]:
        state = self._require_state()
        return {
            "rgb": self.render(),
            "state": state.vector(),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        state = self._require_state()
        if state.collision or state.success or state.step_count >= self.config.horizon:
            raise RuntimeError("cannot step a terminated episode")
        requested = np.asarray(action, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(requested))
        executed = requested
        if norm > self.config.max_step:
            executed = requested * (self.config.max_step / norm)
        low = self.config.workspace_low + self.config.agent_radius
        high = self.config.workspace_high - self.config.agent_radius
        next_position = np.clip(state.position + executed, low, high)
        executed = next_position - state.position
        clearance = swept_clearance(
            state.position,
            next_position,
            state.layout.obstacles,
            self.config.agent_radius,
        )
        state.position = next_position
        state.step_count += 1
        state.path_length += float(np.linalg.norm(executed))
        state.minimum_clearance = min(state.minimum_clearance, clearance)
        state.collision = clearance <= 0.0
        state.success = (
            not state.collision
            and float(np.linalg.norm(state.position - np.asarray(state.layout.goal)))
            <= self.config.goal_radius
        )
        timeout = (
            state.step_count >= self.config.horizon
            and not state.collision
            and not state.success
        )
        done = state.collision or state.success or timeout
        info = {
            "collision": state.collision,
            "success": state.success,
            "timeout": timeout,
            "executed_action": executed.astype(np.float32),
            "minimum_clearance": state.minimum_clearance,
            "path_length": state.path_length,
        }
        return self.observation(), float(state.success), done, info

    def clone_state(self) -> PointNavState:
        return self._require_state().copy()

    def restore_state(self, state: PointNavState) -> dict[str, np.ndarray]:
        self.state = state.copy()
        self.layout = state.layout
        self.difficulty = state.layout.difficulty
        return self.observation()

    def render(self) -> np.ndarray:
        state = self._require_state()
        size = self.config.image_size
        image = np.full((size, size, 3), 242, dtype=np.uint8)
        yy, xx = np.mgrid[0:size, 0:size]

        def pixels(point: tuple[float, float] | np.ndarray) -> tuple[float, float]:
            x, y = float(point[0]), float(point[1])
            px = (x - self.config.workspace_low) / (
                self.config.workspace_high - self.config.workspace_low
            ) * (size - 1)
            py = (self.config.workspace_high - y) / (
                self.config.workspace_high - self.config.workspace_low
            ) * (size - 1)
            return px, py

        def circle(point: tuple[float, float] | np.ndarray, radius: float, color: tuple[int, int, int]) -> None:
            px, py = pixels(point)
            radius_px = radius / (
                self.config.workspace_high - self.config.workspace_low
            ) * (size - 1)
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius_px**2
            image[mask] = color

        circle(state.layout.goal, self.config.goal_radius, (75, 180, 90))
        for obstacle in state.layout.obstacles:
            circle((obstacle.x, obstacle.y), obstacle.radius, (205, 80, 72))
        circle(state.position, self.config.agent_radius, (55, 95, 210))
        return image


def write_layout_manifest(
    path: str | Path,
    *,
    split_seeds: dict[str, dict[int, list[int]]],
    config: EnvironmentConfig | None = None,
) -> dict[str, object]:
    """Freeze named layout splits and their content hash."""

    cfg = config or EnvironmentConfig()
    splits: dict[str, list[dict[str, object]]] = {}
    for split, seeds_by_difficulty in split_seeds.items():
        layouts = [
            make_layout(difficulty, seed, cfg).to_dict()
            for difficulty in sorted(seeds_by_difficulty)
            for seed in seeds_by_difficulty[difficulty]
        ]
        splits[split] = layouts
    payload: dict[str, object] = {
        "version": 1,
        "environment": asdict(cfg),
        "splits": splits,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_layout_manifest(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = payload.pop("sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != claimed:
        raise ValueError("layout manifest hash mismatch")
    payload["sha256"] = claimed
    return payload
