"""Minimal executable versions of MPD's Empty2D geometry environments.

Geometry names, workspace limits, square size, and sphere radius are derived
from ``env_empty_2d.py`` in mpd-splines-public at commit
3676cbf371b0e2a98663d8cef666f671e004e3da. The upstream MIT license is kept
in ``MPD_LICENSE.txt``. No upstream SDF, planner, spline, or model code is
included here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


WORKSPACE_LOW = np.asarray([-1.0, -1.0], dtype=np.float64)
WORKSPACE_HIGH = np.asarray([1.0, 1.0], dtype=np.float64)
START = np.asarray([-0.875, 0.0], dtype=np.float64)
GOAL = np.asarray([0.875, 0.0], dtype=np.float64)


def _segment_point_distance(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator == 0.0:
        return float(np.linalg.norm(start - point))
    fraction = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(start + fraction * delta - point))


def _segment_intersects_box(
    start: np.ndarray,
    end: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> bool:
    """Liang-Barsky slab test for a closed axis-aligned box."""

    direction = end - start
    near, far = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1e-15:
            if start[axis] < low[axis] or start[axis] > high[axis]:
                return False
            continue
        first = (low[axis] - start[axis]) / direction[axis]
        second = (high[axis] - start[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        near = max(near, float(first))
        far = min(far, float(second))
        if near > far:
            return False
    return True


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Distance between disjoint 2-D segments via their four endpoints."""

    return min(
        _segment_point_distance(first_start, first_end, second_start),
        _segment_point_distance(first_start, first_end, second_end),
        _segment_point_distance(second_start, second_end, first_start),
        _segment_point_distance(second_start, second_end, first_end),
    )


@dataclass(frozen=True)
class Empty2DState:
    position: np.ndarray
    step_count: int
    collision_any: bool
    minimum_clearance: float
    clip_count: int


class EnvEmpty2D:
    """Point-mass environment with absolute target actions and swept checks."""

    obstacle_kind: str | None = None
    obstacle_radius = 0.0
    square_half_extent = 0.0

    def __init__(
        self,
        *,
        max_step: float = 0.03,
        robot_radius: float = 0.02,
        goal_tolerance: float = 0.03,
        horizon: int = 320,
    ):
        if max_step <= 0 or robot_radius < 0 or goal_tolerance <= 0 or horizon <= 0:
            raise ValueError("Environment scales and horizon must be positive")
        self.max_step = float(max_step)
        self.robot_radius = float(robot_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.horizon = int(horizon)
        self.position = START.copy()
        self.step_count = 0
        self.collision_any = False
        self.minimum_clearance = float("inf")
        self.clip_count = 0
        self.trajectory = [self.position.copy()]

    def _observation(self) -> dict[str, np.ndarray]:
        return {"agent_pos": self.position.astype(np.float32, copy=True)}

    def reset(self) -> dict[str, np.ndarray]:
        self.position = START.copy()
        self.step_count = 0
        self.collision_any = False
        self.minimum_clearance = float("inf")
        self.clip_count = 0
        self.trajectory = [self.position.copy()]
        return self._observation()

    def get_state(self) -> dict[str, Any]:
        return {
            "position": self.position.copy(),
            "step_count": self.step_count,
            "collision_any": self.collision_any,
            "minimum_clearance": self.minimum_clearance,
            "clip_count": self.clip_count,
            "trajectory": np.asarray(self.trajectory, dtype=np.float64),
        }

    def reset_to(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        position = np.asarray(state["position"], dtype=np.float64)
        if position.shape != (2,) or np.any(position < WORKSPACE_LOW) or np.any(position > WORKSPACE_HIGH):
            raise ValueError("state position must lie in the 2-D workspace")
        self.position = position.copy()
        self.step_count = int(state.get("step_count", 0))
        self.collision_any = bool(state.get("collision_any", False))
        self.minimum_clearance = float(state.get("minimum_clearance", float("inf")))
        self.clip_count = int(state.get("clip_count", 0))
        trajectory = state.get("trajectory", position[None])
        self.trajectory = [point.copy() for point in np.asarray(trajectory, dtype=np.float64)]
        return self._observation()

    def _point_clearance(self, point: np.ndarray) -> float:
        if self.obstacle_kind is None:
            return float("inf")
        if self.obstacle_kind == "circle":
            return float(np.linalg.norm(point) - self.obstacle_radius - self.robot_radius)
        delta = np.abs(point) - self.square_half_extent
        outside = np.maximum(delta, 0.0)
        signed = float(np.linalg.norm(outside) + min(max(delta[0], delta[1]), 0.0))
        return signed - self.robot_radius

    def segment_collides(self, start: np.ndarray, end: np.ndarray) -> bool:
        if self.obstacle_kind is None:
            return False
        if self.obstacle_kind == "circle":
            return _segment_point_distance(start, end, np.zeros(2)) <= (
                self.obstacle_radius + self.robot_radius
            )
        expanded = self.square_half_extent + self.robot_radius
        return _segment_intersects_box(
            start,
            end,
            np.asarray([-expanded, -expanded]),
            np.asarray([expanded, expanded]),
        )

    def segment_clearance(self, start: np.ndarray, end: np.ndarray) -> float:
        """Exact non-negative swept clearance for the circular robot body."""

        if self.obstacle_kind is None:
            return float("inf")
        if self.segment_collides(start, end):
            return 0.0
        if self.obstacle_kind == "circle":
            return _segment_point_distance(start, end, np.zeros(2)) - (
                self.obstacle_radius + self.robot_radius
            )
        extent = self.square_half_extent
        corners = (
            np.asarray([-extent, -extent]),
            np.asarray([extent, -extent]),
            np.asarray([extent, extent]),
            np.asarray([-extent, extent]),
        )
        distance = min(
            _segment_segment_distance(start, end, corners[index], corners[(index + 1) % 4])
            for index in range(4)
        )
        return distance - self.robot_radius

    def trajectory_collides(self, points: np.ndarray) -> bool:
        points = np.asarray(points, dtype=np.float64)
        return any(self.segment_collides(a, b) for a, b in zip(points[:-1], points[1:]))

    def step(self, absolute_xy: np.ndarray):
        target = np.asarray(absolute_xy, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("absolute_xy must be a finite shape-[2] action")
        displacement = target - self.position
        distance = float(np.linalg.norm(displacement))
        clipped = distance > self.max_step
        if clipped:
            displacement *= self.max_step / distance
        executed = np.clip(self.position + displacement, WORKSPACE_LOW, WORKSPACE_HIGH)
        workspace_clipped = not np.array_equal(executed, self.position + displacement)
        if clipped or workspace_clipped:
            self.clip_count += 1
        collision = self.segment_collides(self.position, executed)
        self.minimum_clearance = min(
            self.minimum_clearance,
            self.segment_clearance(self.position, executed),
        )
        self.collision_any = self.collision_any or collision
        self.position = executed
        self.step_count += 1
        self.trajectory.append(self.position.copy())
        success = self.is_success()
        truncated = self.step_count >= self.horizon and not success
        reward = float(success)
        info = {
            "success": success,
            "collision": collision,
            "collision_any": self.collision_any,
            "safe_success": success and not self.collision_any,
            "ncr": truncated and not self.collision_any,
            "truncated": truncated,
            "action_clipped": clipped or workspace_clipped,
            "minimum_clearance": self.minimum_clearance,
        }
        return self._observation(), reward, success or truncated, info

    def is_success(self) -> bool:
        return bool(np.linalg.norm(self.position - GOAL) <= self.goal_tolerance)

    def oracle_obstacle_points(self) -> np.ndarray:
        return np.empty((0, 2), dtype=np.float32)

    def render(
        self,
        *,
        trajectory: np.ndarray | None = None,
        predicted_chunk: np.ndarray | None = None,
        closest_obstacle_point: np.ndarray | None = None,
        safety_distance: float | None = None,
        size: int = 512,
    ) -> np.ndarray:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Circle, Rectangle

        figure, axis = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
        FigureCanvasAgg(figure)
        axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
        axis.grid(alpha=0.15)
        if self.obstacle_kind == "circle":
            axis.add_patch(Circle((0, 0), self.obstacle_radius, color="tab:red", alpha=0.35))
        elif self.obstacle_kind == "square":
            extent = self.square_half_extent
            axis.add_patch(Rectangle((-extent, -extent), 2 * extent, 2 * extent, color="tab:red", alpha=0.35))
        axis.scatter(*START, marker="o", color="black", label="start")
        axis.scatter(*GOAL, marker="*", s=110, color="gold", edgecolor="black", label="goal")
        trace = np.asarray(self.trajectory if trajectory is None else trajectory)
        if len(trace):
            axis.plot(trace[:, 0], trace[:, 1], color="tab:blue", linewidth=2)
            axis.add_patch(
                Circle(
                    trace[-1],
                    self.robot_radius,
                    facecolor="tab:blue",
                    edgecolor="navy",
                    linewidth=1.0,
                    zorder=5,
                )
            )
        if predicted_chunk is not None and len(predicted_chunk):
            chunk = np.asarray(predicted_chunk)
            axis.plot(chunk[:, 0], chunk[:, 1], "--", color="tab:orange", linewidth=1.5)
        if closest_obstacle_point is not None:
            point = np.asarray(closest_obstacle_point)
            axis.scatter(*point[:2], marker="x", color="purple", s=60)
            if safety_distance is not None:
                axis.add_patch(Circle(point[:2], safety_distance, fill=False, linestyle=":", color="purple"))
        figure.tight_layout()
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        image = rgba[..., :3].copy()
        plt.close(figure)
        return image


class EnvEmpty2DExtraSphere(EnvEmpty2D):
    obstacle_kind = "circle"
    obstacle_radius = 0.3

    def oracle_obstacle_points(self) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32) * self.obstacle_radius


class EnvEmpty2DExtraSquare(EnvEmpty2D):
    obstacle_kind = "square"
    square_half_extent = 0.3

    def oracle_obstacle_points(self) -> np.ndarray:
        values = np.linspace(-self.square_half_extent, self.square_half_extent, 64, endpoint=False)
        top = np.column_stack((values, np.full(64, self.square_half_extent)))
        right = np.column_stack((np.full(64, self.square_half_extent), values[::-1]))
        bottom = np.column_stack((values[::-1], np.full(64, -self.square_half_extent)))
        left = np.column_stack((np.full(64, -self.square_half_extent), values))
        return np.concatenate((top, right, bottom, left), axis=0).astype(np.float32)
