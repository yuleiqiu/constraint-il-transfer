"""Obstacle-free Tabletop teacher with broad, latent transport-route support.

The transport path is a straight-line mean plus a scalar endpoint-conditioned
RBF Gaussian-process residual in the path-normal direction. Obstacles are not
present and never enter route generation. This mirrors the Empty2D data
contract: broad route coverage is generated first and counterfactual obstacle
support is measured only afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from robosuite.utils import transform_utils as T


DIRECT_TEACHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-14_tabletop_guidance"
    / "waypoint_teacher.py"
)
_SPEC = importlib.util.spec_from_file_location("tabletop_direct_teacher", DIRECT_TEACHER_PATH)
DIRECT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = DIRECT
_SPEC.loader.exec_module(DIRECT)

TransitionCallback = Callable[..., None]


@dataclass(frozen=True)
class WideSupportTeacherConfig(DIRECT.TeacherConfig):
    route_points: int = 129
    route_length_scale: float = 1.1
    route_kernel_amplitude_m2: float = 0.4
    route_max_lateral_m: float = 0.16
    route_max_attempts: int = 256


@dataclass
class WideSupportTeacherResult:
    success: bool
    steps: int
    grasped: bool
    phases: list[str] = field(default_factory=list)
    can_start: np.ndarray | None = None
    goal_xy: np.ndarray | None = None
    max_transport_line_deviation_m: float = 0.0
    commanded_route_signed_peak_m: float = 0.0
    commanded_route_midpoint_offset_m: float = 0.0
    achieved_route_signed_peak_m: float = 0.0
    route_draw_attempts: int = 0


def conditional_rbf_covariance(num_points, length_scale, amplitude_m2):
    """Return endpoint-conditioned scalar RBF covariance in square metres."""

    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if length_scale <= 0 or amplitude_m2 <= 0:
        raise ValueError("length_scale and amplitude_m2 must be positive")
    time = np.linspace(0.0, 1.0, int(num_points))
    squared_distance = (time[:, None] - time[None, :]) ** 2
    kernel = float(amplitude_m2) * np.exp(
        -0.5 * squared_distance / float(length_scale) ** 2
    )
    endpoints = np.asarray([0, len(time) - 1])
    endpoint_kernel = kernel[np.ix_(endpoints, endpoints)]
    conditioned = kernel - kernel[:, endpoints] @ np.linalg.solve(
        endpoint_kernel, kernel[endpoints]
    )
    conditioned = 0.5 * (conditioned + conditioned.T)
    return conditioned + 1e-12 * np.eye(len(time))


def sample_lateral_route(start, goal, rng, config):
    """Sample a bounded, endpoint-exact XY route without obstacle knowledge."""

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start and goal must have shape (3,)")
    delta_xy = goal[:2] - start[:2]
    distance_xy = float(np.linalg.norm(delta_xy))
    if distance_xy <= 1e-8:
        raise ValueError("transport start and goal must differ in XY")
    tangent = delta_xy / distance_xy
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    covariance = conditional_rbf_covariance(
        config.route_points,
        config.route_length_scale,
        config.route_kernel_amplitude_m2,
    )
    cholesky = np.linalg.cholesky(covariance)
    fraction = np.linspace(0.0, 1.0, config.route_points)
    mean = start[None] + fraction[:, None] * (goal - start)[None]
    for attempt in range(1, config.route_max_attempts + 1):
        lateral = cholesky @ rng.standard_normal(config.route_points)
        lateral[[0, -1]] = 0.0
        signed_peak = float(lateral[np.argmax(np.abs(lateral))])
        if abs(signed_peak) > config.route_max_lateral_m:
            continue
        route = mean.copy()
        route[:, :2] += lateral[:, None] * normal[None]
        midpoint_offset = float(lateral[len(lateral) // 2])
        return route, normal, signed_peak, midpoint_offset, attempt
    raise RuntimeError("could not sample a bounded lateral transport route")


def resample_polyline_max_step(points, max_step_m):
    """Densify a polyline so every commanded Cartesian displacement is bounded."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("points must have shape [N>=2, 3]")
    if max_step_m <= 0:
        raise ValueError("max_step_m must be positive")
    output = [points[0].copy()]
    for start, goal in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(goal - start))
        count = max(1, int(np.ceil(distance / max_step_m)))
        for index in range(1, count + 1):
            output.append(start + (index / count) * (goal - start))
    return np.asarray(output)


class WideSupportPickPlaceTeacher:
    """Closed-loop obstacle-free teacher with broad symmetric transport routes."""

    def __init__(self, config: WideSupportTeacherConfig | None = None):
        self.config = config or WideSupportTeacherConfig()

    def rollout(self, env, seed: int, transition_callback: TransitionCallback | None = None):
        rng = np.random.default_rng(seed)
        obs = env.reset()
        raw_env = getattr(env, "env", env)
        if raw_env.has_obstacle:
            raise ValueError("WideSupportPickPlaceTeacher requires the obstacle-free source environment")

        can_start = np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id], dtype=np.float64).copy()
        goal_xy = np.asarray(raw_env.goal_xy, dtype=np.float64).copy()
        fixed_rotation = T.quat2axisangle(np.asarray(obs["robot0_eef_quat_site"]).copy())
        command_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        speed_scale = rng.uniform(*self.config.speed_scale_range)
        max_step = self.config.max_cartesian_step_m * speed_scale
        height_jitter = rng.uniform(-self.config.height_jitter_m, self.config.height_jitter_m)
        grasp_xy = rng.uniform(-self.config.grasp_xy_jitter_m, self.config.grasp_xy_jitter_m, size=2)

        steps = 0
        phases = []
        grasped = False
        transport_positions = []
        transport_start = None
        transport_goal = None
        route_normal = None
        route_signed_peak = 0.0
        route_midpoint = 0.0
        route_attempts = 0

        def step(action, phase):
            nonlocal obs, steps, grasped
            if steps >= self.config.maximum_steps:
                return False
            obs_before = {key: np.asarray(value).copy() for key, value in obs.items()}
            state_before = (
                np.asarray(raw_env.sim.get_state().flatten()).copy()
                if transition_callback is not None
                else None
            )
            next_obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
            steps += 1
            phases.append(phase)
            grasped |= bool(raw_env._check_grasp(raw_env.robots[0].gripper, raw_env.can))
            if phase == "transport":
                transport_positions.append(np.asarray(next_obs["robot0_eef_pos"], dtype=np.float64).copy())
            success = bool(raw_env._check_success())
            if transition_callback is not None:
                transition_callback(
                    state_before=state_before,
                    obs=obs_before,
                    next_obs={key: np.asarray(value).copy() for key, value in next_obs.items()},
                    controller_action=np.asarray(action, dtype=np.float32).copy(),
                    reward=float(reward),
                    done=bool(done),
                    success=success,
                    phase=phase,
                )
            obs = next_obs
            return not (done or success or steps >= self.config.maximum_steps)

        def action(position, gripper):
            return np.concatenate((np.asarray(position), fixed_rotation, [float(gripper)]))

        def hold(gripper, count, phase):
            for _ in range(count):
                if not step(action(command_pos, gripper), phase):
                    return False
            return True

        def move(target, gripper, phase):
            nonlocal command_pos
            start = command_pos.copy()
            for position in DIRECT.minimum_jerk_positions(start, target, max_step):
                command_pos = position.copy()
                if not step(action(command_pos, gripper), phase):
                    return False
            return hold(gripper, DIRECT._sample_integer(rng, self.config.settle_steps_range), f"{phase}_settle")

        def result():
            achieved_peak = 0.0
            deviation = 0.0
            if transport_positions and transport_start is not None and transport_goal is not None:
                positions = np.asarray(transport_positions)
                deviation = float(
                    np.max(DIRECT.point_to_segment_distance(positions[:, :2], transport_start, transport_goal))
                )
                if route_normal is not None:
                    tangent = transport_goal - transport_start
                    denom = float(tangent @ tangent)
                    alpha = np.clip(((positions[:, :2] - transport_start) @ tangent) / denom, 0.0, 1.0)
                    line = transport_start + alpha[:, None] * tangent
                    signed = (positions[:, :2] - line) @ route_normal
                    achieved_peak = float(signed[np.argmax(np.abs(signed))])
            return WideSupportTeacherResult(
                success=bool(raw_env._check_success()),
                steps=steps,
                grasped=grasped,
                phases=phases,
                can_start=can_start,
                goal_xy=goal_xy,
                max_transport_line_deviation_m=deviation,
                commanded_route_signed_peak_m=route_signed_peak,
                commanded_route_midpoint_offset_m=route_midpoint,
                achieved_route_signed_peak_m=achieved_peak,
                route_draw_attempts=route_attempts,
            )

        if not hold(-1.0, DIRECT._sample_integer(rng, self.config.open_steps_range), "open"):
            return result()

        can_pos = np.asarray(raw_env.sim.data.body_xpos[raw_env.can_body_id], dtype=np.float64).copy()
        grasp_pos = can_pos + np.array([grasp_xy[0], grasp_xy[1], self.config.grasp_z_offset_m])
        pregrasp_pos = grasp_pos.copy()
        pregrasp_pos[2] = can_pos[2] + self.config.pregrasp_height_m + height_jitter
        lift_pos = grasp_pos.copy()
        lift_pos[2] = self.config.transport_eef_z_m + height_jitter
        place_pos = np.array(
            [goal_xy[0] + 0.25 * grasp_xy[0], goal_xy[1] + 0.25 * grasp_xy[1], self.config.placement_eef_z_m],
            dtype=np.float64,
        )
        transport_goal_pos = np.array([place_pos[0], place_pos[1], lift_pos[2]], dtype=np.float64)
        retreat_pos = place_pos.copy()
        retreat_pos[2] += self.config.retreat_height_m + max(0.0, height_jitter)

        for target, gripper, phase in (
            (pregrasp_pos, -1.0, "approach"),
            (grasp_pos, -1.0, "descend"),
        ):
            if not move(target, gripper, phase):
                return result()
        if not hold(1.0, DIRECT._sample_integer(rng, self.config.close_steps_range), "close"):
            return result()
        if not move(lift_pos, 1.0, "lift"):
            return result()

        transport_start = lift_pos[:2].copy()
        transport_goal = transport_goal_pos[:2].copy()
        route, route_normal, route_signed_peak, route_midpoint, route_attempts = sample_lateral_route(
            lift_pos, transport_goal_pos, rng, self.config
        )
        route = resample_polyline_max_step(route, max_step)
        for position in route[1:]:
            command_pos = position.copy()
            if not step(action(command_pos, 1.0), "transport"):
                return result()
        if not hold(1.0, DIRECT._sample_integer(rng, self.config.settle_steps_range), "transport_settle"):
            return result()
        if not move(place_pos, 1.0, "lower"):
            return result()
        if not hold(-1.0, DIRECT._sample_integer(rng, self.config.release_steps_range), "release"):
            return result()
        move(retreat_pos, -1.0, "retreat")
        if not raw_env._check_success() and steps < self.config.maximum_steps:
            hold(-1.0, self.config.final_hold_steps, "final_hold")
        return result()


absolute_osc_controller_config = DIRECT.absolute_osc_controller_config
