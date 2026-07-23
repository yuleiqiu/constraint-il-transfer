"""Eight-step task-constrained delta-EEF feasibility correction.

This module deliberately contains no policy or simulator dependencies. It is
the Stage-1 upper-bound solver described in
``docs/guided_denoising/task_constrained_posterior_guidance_plan.md``.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class FeasibilityConfig:
    clearance_margin_m: float = 0.0
    clearance_recovery_steps: int = 0
    progress_tolerance_m: float = 5e-4
    trust_radius_m: float = 0.02
    smoothness_weight: float = 1.0
    multistart_amplitude_fraction: float = 0.75
    multistart_directions: int = 8
    solver_max_iterations: int = 500
    solver_ftol: float = 1e-10
    constraint_tolerance: float = 1e-5

    def validate(self):
        if self.clearance_margin_m < 0:
            raise ValueError("clearance_margin_m must be non-negative")
        if self.clearance_recovery_steps < 0:
            raise ValueError("clearance_recovery_steps must be non-negative")
        if self.progress_tolerance_m < 0:
            raise ValueError("progress_tolerance_m must be non-negative")
        if self.trust_radius_m <= 0:
            raise ValueError("trust_radius_m must be positive")
        if self.smoothness_weight < 0:
            raise ValueError("smoothness_weight must be non-negative")
        if not 0 < self.multistart_amplitude_fraction <= 1:
            raise ValueError("multistart_amplitude_fraction must be in (0, 1]")
        if self.multistart_directions < 0:
            raise ValueError("multistart_directions must be non-negative")
        if self.solver_max_iterations <= 0:
            raise ValueError("solver_max_iterations must be positive")
        if self.solver_ftol <= 0 or self.constraint_tolerance <= 0:
            raise ValueError("solver and constraint tolerances must be positive")


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    corrected_actions: np.ndarray
    baseline_positions: np.ndarray
    corrected_positions: np.ndarray
    waypoint_offsets_xy: np.ndarray
    objective: float
    minimum_constraint_value: float
    baseline_min_clearance_m: float
    corrected_min_clearance_m: float
    baseline_target_progress_m: float
    corrected_target_progress_m: float
    max_waypoint_displacement_m: float
    action_bound_violation_max: float
    solver_success: bool
    solver_status: int
    solver_message: str
    solver_iterations: int
    selected_start: int
    starts_attempted: int

    def to_dict(self):
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            elif isinstance(value, np.generic):
                payload[key] = value.item()
        return payload


def _array(value, *, ndim, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError("{} must have {} dimensions".format(name, ndim))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} must be finite".format(name))
    return result


def reconstruct_positions(start_position, actions):
    """Reconstruct absolute EEF positions from raw delta-EEF actions."""

    start = _array(start_position, ndim=1, name="start_position")
    action_array = _array(actions, ndim=2, name="actions")
    if start.shape != (3,) or action_array.shape[1] < 3:
        raise ValueError("Expected a 3-D start and actions with at least 3 dimensions")
    return start[None] + np.cumsum(action_array[:, :3], axis=0)


def positions_to_actions(start_position, positions, template_actions):
    """Replace only XYZ deltas while preserving all other action dimensions."""

    start = _array(start_position, ndim=1, name="start_position")
    position_array = _array(positions, ndim=2, name="positions")
    template = _array(template_actions, ndim=2, name="template_actions")
    if start.shape != (3,) or position_array.shape != (template.shape[0], 3):
        raise ValueError("Position horizon must match the action horizon")
    with_start = np.concatenate((start[None], position_array), axis=0)
    corrected = template.copy()
    corrected[:, :3] = np.diff(with_start, axis=0)
    return corrected


def closest_target_progress(start_position, positions, target_position):
    """Positive progress is reduction in the closest 3-D target distance."""

    start = _array(start_position, ndim=1, name="start_position")
    points = _array(positions, ndim=2, name="positions")
    target = _array(target_position, ndim=1, name="target_position")
    if start.shape != (3,) or target.shape != (3,) or points.shape[1] != 3:
        raise ValueError("Target progress requires 3-D positions")
    start_distance = np.linalg.norm(start - target)
    closest_distance = np.min(np.linalg.norm(points - target[None], axis=1))
    return float(start_distance - closest_distance)


def segment_circle_clearances_xy(start_position, positions, centers, radii):
    """Signed clearance for every EEF segment and circular XY obstacle."""

    start = _array(start_position, ndim=1, name="start_position")
    points = _array(positions, ndim=2, name="positions")
    center_array = _array(centers, ndim=2, name="centers")
    radius_array = _array(radii, ndim=1, name="radii")
    if start.shape != (3,) or points.shape[1] != 3:
        raise ValueError("EEF path positions must be 3-D")
    if center_array.shape[1] not in (2, 3):
        raise ValueError("Obstacle centers must have shape [N, 2] or [N, 3]")
    if center_array.shape[0] != radius_array.shape[0]:
        raise ValueError("Obstacle center and radius counts must match")
    if np.any(radius_array < 0):
        raise ValueError("Obstacle radii must be non-negative")
    if center_array.shape[0] == 0:
        return np.empty((points.shape[0], 0), dtype=np.float64)

    polyline = np.concatenate((start[None, :2], points[:, :2]), axis=0)
    segment_start = polyline[:-1, None, :]
    segment_delta = (polyline[1:] - polyline[:-1])[:, None, :]
    center_delta = center_array[None, :, :2] - segment_start
    denominator = np.sum(segment_delta * segment_delta, axis=-1)
    numerator = np.sum(center_delta * segment_delta, axis=-1)
    fraction = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-16,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = segment_start + fraction[..., None] * segment_delta
    distances = np.linalg.norm(closest - center_array[None, :, :2], axis=-1)
    return distances - radius_array[None]


def minimum_segment_clearance_xy(start_position, positions, centers, radii):
    clearances = segment_circle_clearances_xy(start_position, positions, centers, radii)
    return float(np.min(clearances)) if clearances.size else float("inf")


def point_circle_clearances_xy(position, centers, radii):
    point = _array(position, ndim=1, name="position")
    center_array = _array(centers, ndim=2, name="centers")
    radius_array = _array(radii, ndim=1, name="radii")
    if point.shape != (3,) or center_array.shape[1] not in (2, 3):
        raise ValueError("Point and obstacle positions have incompatible shapes")
    if center_array.shape[0] != radius_array.shape[0]:
        raise ValueError("Obstacle center and radius counts must match")
    if center_array.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(center_array[:, :2] - point[None, :2], axis=1) - radius_array


def _smoothness(offsets):
    if offsets.shape[0] < 3:
        return 0.0
    second_difference = offsets[2:] - 2.0 * offsets[1:-1] + offsets[:-2]
    return float(np.sum(second_difference * second_difference))


def _initial_offsets(horizon, config):
    yield np.zeros((horizon, 2), dtype=np.float64)
    if config.multistart_directions == 0:
        return
    phase = np.sin(np.pi * np.arange(1, horizon + 1) / (horizon + 1))[:, None]
    amplitude = config.trust_radius_m * config.multistart_amplitude_fraction
    for index in range(config.multistart_directions):
        angle = 2.0 * np.pi * index / config.multistart_directions
        direction = np.array((np.cos(angle), np.sin(angle)), dtype=np.float64)
        yield amplitude * phase * direction[None]


def solve_eight_step_feasibility(
    *,
    start_position,
    baseline_actions,
    target_position,
    obstacle_centers,
    obstacle_radii,
    action_low,
    action_high,
    config=FeasibilityConfig(),
):
    """Find the smallest task-preserving XY correction to one action chunk."""

    config.validate()
    start = _array(start_position, ndim=1, name="start_position")
    baseline = _array(baseline_actions, ndim=2, name="baseline_actions")
    target = _array(target_position, ndim=1, name="target_position")
    centers = _array(obstacle_centers, ndim=2, name="obstacle_centers")
    radii = _array(obstacle_radii, ndim=1, name="obstacle_radii")
    low = _array(action_low, ndim=1, name="action_low")
    high = _array(action_high, ndim=1, name="action_high")
    if baseline.shape[0] != 8:
        raise ValueError("Stage-1 feasibility requires the frozen eight-action horizon")
    if baseline.shape[1] < 3 or low.shape != (baseline.shape[1],) or high.shape != low.shape:
        raise ValueError("Action limits must match the action dimension")
    if np.any(low > high):
        raise ValueError("Action lower bounds must not exceed upper bounds")

    baseline_positions = reconstruct_positions(start, baseline)
    baseline_progress = closest_target_progress(start, baseline_positions, target)
    baseline_clearance = minimum_segment_clearance_xy(start, baseline_positions, centers, radii)
    start_clearances = point_circle_clearances_xy(start, centers, radii)
    horizon = baseline.shape[0]
    waypoint_weights = np.linspace(2.0, 1.0, horizon)[:, None]

    def unpack(flat_offsets):
        offsets = np.asarray(flat_offsets, dtype=np.float64).reshape(horizon, 2)
        positions = baseline_positions.copy()
        positions[:, :2] += offsets
        actions = positions_to_actions(start, positions, baseline)
        return offsets, positions, actions

    def objective(flat_offsets):
        offsets, _, _ = unpack(flat_offsets)
        deviation = np.sum(waypoint_weights * offsets * offsets)
        return float(deviation + config.smoothness_weight * _smoothness(offsets))

    def constraint_values(flat_offsets):
        offsets, positions, actions = unpack(flat_offsets)
        values = []
        clearances = segment_circle_clearances_xy(start, positions, centers, radii)
        if clearances.size:
            requirements = np.full_like(clearances, config.clearance_margin_m)
            if config.clearance_recovery_steps > 0:
                fractions = np.minimum(
                    np.arange(horizon, dtype=np.float64)
                    / float(config.clearance_recovery_steps),
                    1.0,
                )[:, None]
                unsafe_start = start_clearances < config.clearance_margin_m
                requirements[:, unsafe_start] = (
                    start_clearances[None, unsafe_start]
                    + fractions
                    * (
                        config.clearance_margin_m
                        - start_clearances[None, unsafe_start]
                    )
                )
            values.extend((clearances - requirements).reshape(-1))
        progress = closest_target_progress(start, positions, target)
        values.append(progress - baseline_progress + config.progress_tolerance_m)
        values.extend(config.trust_radius_m**2 - np.sum(offsets * offsets, axis=1))
        values.extend((actions - low[None]).reshape(-1))
        values.extend((high[None] - actions).reshape(-1))
        return np.asarray(values, dtype=np.float64)

    constraints = ({"type": "ineq", "fun": constraint_values},)
    component_bound = config.trust_radius_m
    bounds = [(-component_bound, component_bound)] * (horizon * 2)
    candidates = []
    initial_guesses = list(_initial_offsets(horizon, config))
    for start_index, initial in enumerate(initial_guesses):
        result = minimize(
            objective,
            initial.reshape(-1),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": config.solver_max_iterations,
                "ftol": config.solver_ftol,
                "disp": False,
            },
        )
        values = constraint_values(result.x)
        minimum_value = float(np.min(values))
        feasible = bool(minimum_value >= -config.constraint_tolerance)
        candidates.append((feasible, objective(result.x), -minimum_value, start_index, result))

    candidates.sort(key=lambda item: (not item[0], item[1] if item[0] else item[2], item[3]))
    feasible, selected_objective, _, selected_start, selected = candidates[0]
    offsets, corrected_positions, corrected_actions = unpack(selected.x)
    corrected_clearance = minimum_segment_clearance_xy(
        start, corrected_positions, centers, radii
    )
    corrected_progress = closest_target_progress(start, corrected_positions, target)
    lower_violation = np.maximum(low[None] - corrected_actions, 0.0)
    upper_violation = np.maximum(corrected_actions - high[None], 0.0)
    action_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
    minimum_value = float(np.min(constraint_values(selected.x)))

    return FeasibilityResult(
        feasible=feasible,
        corrected_actions=corrected_actions,
        baseline_positions=baseline_positions,
        corrected_positions=corrected_positions,
        waypoint_offsets_xy=offsets,
        objective=float(selected_objective),
        minimum_constraint_value=minimum_value,
        baseline_min_clearance_m=baseline_clearance,
        corrected_min_clearance_m=corrected_clearance,
        baseline_target_progress_m=baseline_progress,
        corrected_target_progress_m=corrected_progress,
        max_waypoint_displacement_m=float(np.max(np.linalg.norm(offsets, axis=1))),
        action_bound_violation_max=action_violation,
        solver_success=bool(selected.success),
        solver_status=int(selected.status),
        solver_message=str(selected.message),
        solver_iterations=int(selected.nit),
        selected_start=int(selected_start),
        starts_attempted=len(initial_guesses),
    )
