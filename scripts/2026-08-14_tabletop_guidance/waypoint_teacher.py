"""Scripted Cartesian teacher for obstacle-free PickPlaceCanTabletop demos.

The teacher intentionally contains only the direct pick-and-place task mode.
It uses absolute world-frame OSC_POSE commands and small nuisance variation,
but never inserts a lateral avoidance waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T


TransitionCallback = Callable[..., None]


@dataclass(frozen=True)
class TeacherConfig:
    max_cartesian_step_m: float = 0.006
    pregrasp_height_m: float = 0.12
    grasp_z_offset_m: float = 0.025
    transport_eef_z_m: float = 1.02
    placement_eef_z_m: float = 0.865
    retreat_height_m: float = 0.14
    grasp_xy_jitter_m: float = 0.0025
    height_jitter_m: float = 0.01
    speed_scale_range: tuple[float, float] = (0.9, 1.1)
    open_steps_range: tuple[int, int] = (6, 10)
    close_steps_range: tuple[int, int] = (15, 21)
    release_steps_range: tuple[int, int] = (17, 23)
    settle_steps_range: tuple[int, int] = (2, 5)
    final_hold_steps: int = 4
    maximum_steps: int = 600


@dataclass
class TeacherResult:
    success: bool
    steps: int
    grasped: bool
    phases: list[str] = field(default_factory=list)
    can_start: np.ndarray | None = None
    goal_xy: np.ndarray | None = None
    max_transport_line_deviation_m: float = 0.0


def absolute_osc_controller_config(kp: float = 500.0):
    """Return Panda's OSC config with the project's absolute EEF contract."""

    config = load_composite_controller_config(robot="Panda")
    arm = config["body_parts"]["right"]
    if arm["type"] != "OSC_POSE":
        raise ValueError(f"Panda default arm controller is {arm['type']}, not OSC_POSE")
    arm.update(input_type="absolute", input_ref_frame="world", kp=float(kp))
    return config


def minimum_jerk_positions(start, goal, max_step_m):
    """Return endpoint-inclusive minimum-jerk commands with bounded nominal step."""

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start and goal must both have shape (3,)")
    if max_step_m <= 0:
        raise ValueError("max_step_m must be positive")
    count = max(1, int(np.ceil(np.linalg.norm(goal - start) / max_step_m)))
    u = np.linspace(1.0 / count, 1.0, count)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start[None] + blend[:, None] * (goal - start)[None]


def point_to_segment_distance(points, start, goal):
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    segment = goal - start
    denom = float(segment @ segment)
    if denom <= 1e-12:
        return np.linalg.norm(points - start, axis=-1)
    alpha = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
    projection = start + alpha[..., None] * segment
    return np.linalg.norm(points - projection, axis=-1)


def _sample_integer(rng, bounds):
    low, high = bounds
    if low <= 0 or high < low:
        raise ValueError(f"invalid positive integer range {bounds}")
    return int(rng.integers(low, high + 1))


class DirectPickPlaceTeacher:
    """Closed-loop phase teacher with no obstacle knowledge or detour support."""

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()

    def rollout(self, env, seed: int, transition_callback: TransitionCallback | None = None):
        rng = np.random.default_rng(seed)
        obs = env.reset()
        raw_env = getattr(env, "env", env)
        if raw_env.has_obstacle:
            raise ValueError("DirectPickPlaceTeacher may only collect in the obstacle-free source environment")

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
            for position in minimum_jerk_positions(start, target, max_step):
                command_pos = position.copy()
                if not step(action(command_pos, gripper), phase):
                    return False
            return hold(gripper, _sample_integer(rng, self.config.settle_steps_range), f"{phase}_settle")

        if not hold(-1.0, _sample_integer(rng, self.config.open_steps_range), "open"):
            return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)

        # Re-read the Can after it has physically settled onto the table.
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

        sequence = [
            (pregrasp_pos, -1.0, "approach"),
            (grasp_pos, -1.0, "descend"),
        ]
        for target, gripper, phase in sequence:
            if not move(target, gripper, phase):
                return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)

        if not hold(1.0, _sample_integer(rng, self.config.close_steps_range), "close"):
            return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)
        if not move(lift_pos, 1.0, "lift"):
            return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)

        transport_start = lift_pos[:2].copy()
        transport_goal = transport_goal_pos[:2].copy()
        if not move(transport_goal_pos, 1.0, "transport"):
            return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)
        if not move(place_pos, 1.0, "lower"):
            return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)
        if not hold(-1.0, _sample_integer(rng, self.config.release_steps_range), "release"):
            return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)
        move(retreat_pos, -1.0, "retreat")
        if not raw_env._check_success() and steps < self.config.maximum_steps:
            hold(-1.0, self.config.final_hold_steps, "final_hold")
        return self._result(raw_env, steps, phases, grasped, can_start, goal_xy, transport_positions, transport_start, transport_goal)

    @staticmethod
    def _result(raw_env, steps, phases, grasped, can_start, goal_xy, positions, line_start, line_goal):
        deviation = 0.0
        if positions and line_start is not None and line_goal is not None:
            deviation = float(np.max(point_to_segment_distance(np.asarray(positions)[:, :2], line_start, line_goal)))
        return TeacherResult(
            success=bool(raw_env._check_success()),
            steps=steps,
            grasped=grasped,
            phases=phases,
            can_start=can_start,
            goal_xy=goal_xy,
            max_transport_line_deviation_m=deviation,
        )
