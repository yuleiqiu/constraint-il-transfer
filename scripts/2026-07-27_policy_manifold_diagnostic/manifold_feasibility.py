"""Policy-manifold feasibility for executable delta-EEF action chunks.

This module is intentionally independent of robomimic and MuJoCo. It estimates
a shrinkage covariance over sampled EEF waypoint trajectories, then asks
whether a nominal trajectory can be projected to satisfy geometric constraints
without leaving an empirically calibrated policy-support ellipsoid.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ManifoldConfig:
    shrinkage: float = 0.2
    variance_floor_m: float = 0.001
    support_quantile: float = 0.95
    support_multiplier: float = 1.0
    progress_tolerance_m: float = 5e-4
    solver_max_iterations: int = 500
    solver_ftol: float = 1e-10
    constraint_tolerance: float = 1e-5
    maximum_multistarts: int = 10

    def validate(self):
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if self.variance_floor_m <= 0.0:
            raise ValueError("variance_floor_m must be positive")
        if not 0.5 <= self.support_quantile < 1.0:
            raise ValueError("support_quantile must be in [0.5, 1)")
        if self.support_multiplier <= 0.0:
            raise ValueError("support_multiplier must be positive")
        if self.progress_tolerance_m < 0.0:
            raise ValueError("progress_tolerance_m must be non-negative")
        if self.solver_max_iterations <= 0:
            raise ValueError("solver_max_iterations must be positive")
        if self.solver_ftol <= 0.0 or self.constraint_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.maximum_multistarts < 2:
            raise ValueError("maximum_multistarts must be at least two")


@dataclass(frozen=True)
class ManifoldModel:
    mean_xy: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    support_threshold_d2: float
    sample_distances_d2: np.ndarray
    effective_rank: float
    condition_number: float

    def support_distance_d2(self, positions_xy):
        flat = np.asarray(positions_xy, dtype=np.float64).reshape(-1)
        delta = flat - self.mean_xy
        return float(delta @ self.precision @ delta)

    def to_dict(self):
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ProjectionResult:
    feasible: bool
    corrected_positions: np.ndarray
    corrected_actions: np.ndarray
    objective_d2_from_nominal: float
    support_distance_d2: float
    support_ratio: float
    baseline_min_clearance_m: float
    corrected_min_clearance_m: float
    baseline_target_progress_m: float
    corrected_target_progress_m: float
    maximum_waypoint_displacement_m: float
    minimum_constraint_value: float
    solver_success: bool
    solver_status: int
    solver_message: str
    solver_iterations: int
    selected_start: int
    starts_attempted: int

    def to_dict(self):
        return _jsonable(asdict(self))


def _jsonable(payload):
    result = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _array(value, *, ndim, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError("{} must have {} dimensions".format(name, ndim))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must be finite".format(name))
    return array


def reconstruct_positions(start_position, actions):
    start = _array(start_position, ndim=1, name="start_position")
    chunks = _array(actions, ndim=2, name="actions")
    if start.shape != (3,) or chunks.shape[1] < 3:
        raise ValueError("Expected a 3-D start and actions with XYZ components")
    return start[None] + np.cumsum(chunks[:, :3], axis=0)


def positions_to_actions(start_position, positions, template_actions):
    start = _array(start_position, ndim=1, name="start_position")
    points = _array(positions, ndim=2, name="positions")
    template = _array(template_actions, ndim=2, name="template_actions")
    if start.shape != (3,) or points.shape != (template.shape[0], 3):
        raise ValueError("Position horizon must match the action horizon")
    corrected = template.copy()
    corrected[:, :3] = np.diff(
        np.concatenate((start[None], points), axis=0),
        axis=0,
    )
    return corrected


def closest_target_progress(start_position, positions, target_position):
    start = _array(start_position, ndim=1, name="start_position")
    points = _array(positions, ndim=2, name="positions")
    target = _array(target_position, ndim=1, name="target_position")
    if start.shape != (3,) or target.shape != (3,) or points.shape[1] != 3:
        raise ValueError("Target progress requires 3-D points")
    initial_distance = np.linalg.norm(start - target)
    closest_distance = np.min(np.linalg.norm(points - target[None], axis=1))
    return float(initial_distance - closest_distance)


def segment_circle_clearances_xy(start_position, positions, centers, radii):
    start = _array(start_position, ndim=1, name="start_position")
    points = _array(positions, ndim=2, name="positions")
    obstacle_centers = _array(centers, ndim=2, name="centers")
    obstacle_radii = _array(radii, ndim=1, name="radii")
    if start.shape != (3,) or points.shape[1] != 3:
        raise ValueError("EEF positions must be 3-D")
    if obstacle_centers.shape[1] not in (2, 3):
        raise ValueError("Obstacle centers must contain XY coordinates")
    if len(obstacle_centers) != len(obstacle_radii):
        raise ValueError("Obstacle center and radius counts must match")
    if np.any(obstacle_radii < 0.0):
        raise ValueError("Obstacle radii must be non-negative")
    if len(obstacle_centers) == 0:
        return np.empty((len(points), 0), dtype=np.float64)

    polyline = np.concatenate((start[None, :2], points[:, :2]), axis=0)
    starts = polyline[:-1, None]
    deltas = (polyline[1:] - polyline[:-1])[:, None]
    relative = obstacle_centers[None, :, :2] - starts
    denominator = np.sum(deltas * deltas, axis=-1)
    numerator = np.sum(relative * deltas, axis=-1)
    fraction = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-16,
    )
    closest = starts + np.clip(fraction, 0.0, 1.0)[..., None] * deltas
    return (
        np.linalg.norm(closest - obstacle_centers[None, :, :2], axis=-1)
        - obstacle_radii[None]
    )


def minimum_segment_clearance_xy(start_position, positions, centers, radii):
    values = segment_circle_clearances_xy(
        start_position,
        positions,
        centers,
        radii,
    )
    return float(np.min(values)) if values.size else float("inf")


def fit_manifold(sample_positions, config=ManifoldConfig()):
    """Fit a regularized Gaussian metric to sampled waypoint XY trajectories."""

    config.validate()
    samples = _array(sample_positions, ndim=3, name="sample_positions")
    if samples.shape[0] < 3 or samples.shape[2] != 3:
        raise ValueError("sample_positions must have shape [K>=3, H, 3]")
    flattened = samples[:, :, :2].reshape(samples.shape[0], -1)
    mean = np.mean(flattened, axis=0)
    centered = flattened - mean[None]
    covariance = centered.T @ centered / float(samples.shape[0] - 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - config.shrinkage) * covariance
        + config.shrinkage * diagonal
        + config.variance_floor_m**2 * np.eye(covariance.shape[0])
    )
    precision = np.linalg.inv(covariance)
    distances = np.einsum("ni,ij,nj->n", centered, precision, centered)
    threshold = float(
        np.quantile(distances, config.support_quantile)
        * config.support_multiplier
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    probabilities = eigenvalues / np.sum(eigenvalues)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    return ManifoldModel(
        mean_xy=mean,
        covariance=covariance,
        precision=precision,
        support_threshold_d2=threshold,
        sample_distances_d2=distances,
        effective_rank=effective_rank,
        condition_number=condition_number,
    )


def solve_manifold_projection(
    *,
    start_position,
    nominal_actions,
    sample_positions,
    target_position,
    obstacle_centers,
    obstacle_radii,
    action_low,
    action_high,
    clearance_margin_m,
    config=ManifoldConfig(),
):
    """Find a progress-preserving safe trajectory inside local policy support."""

    config.validate()
    start = _array(start_position, ndim=1, name="start_position")
    nominal = _array(nominal_actions, ndim=2, name="nominal_actions")
    samples = _array(sample_positions, ndim=3, name="sample_positions")
    target = _array(target_position, ndim=1, name="target_position")
    centers = _array(obstacle_centers, ndim=2, name="obstacle_centers")
    radii = _array(obstacle_radii, ndim=1, name="obstacle_radii")
    low = _array(action_low, ndim=1, name="action_low")
    high = _array(action_high, ndim=1, name="action_high")
    if clearance_margin_m < 0.0:
        raise ValueError("clearance_margin_m must be non-negative")
    if samples.shape[1:] != (nominal.shape[0], 3):
        raise ValueError("Sample and nominal horizons must match")
    if low.shape != (nominal.shape[1],) or high.shape != low.shape:
        raise ValueError("Action bounds must match action dimension")
    if np.any(low > high):
        raise ValueError("Action lower bounds must not exceed upper bounds")

    model = fit_manifold(samples, config)
    nominal_positions = reconstruct_positions(start, nominal)
    nominal_xy = nominal_positions[:, :2].reshape(-1)
    nominal_progress = closest_target_progress(start, nominal_positions, target)
    nominal_clearance = minimum_segment_clearance_xy(
        start,
        nominal_positions,
        centers,
        radii,
    )
    z_coordinates = nominal_positions[:, 2].copy()

    def unpack(flat_xy):
        positions = np.column_stack(
            (
                np.asarray(flat_xy, dtype=np.float64).reshape(-1, 2),
                z_coordinates,
            )
        )
        actions = positions_to_actions(start, positions, nominal)
        return positions, actions

    def objective(flat_xy):
        delta = np.asarray(flat_xy, dtype=np.float64) - nominal_xy
        return float(delta @ model.precision @ delta)

    def constraint_values(flat_xy):
        positions, actions = unpack(flat_xy)
        values = []
        clearances = segment_circle_clearances_xy(start, positions, centers, radii)
        if clearances.size:
            values.extend((clearances - clearance_margin_m).reshape(-1))
        progress = closest_target_progress(start, positions, target)
        values.append(
            progress - nominal_progress + config.progress_tolerance_m
        )
        support_d2 = model.support_distance_d2(positions[:, :2])
        values.append(model.support_threshold_d2 - support_d2)
        values.extend((actions[:, :2] - low[None, :2]).reshape(-1))
        values.extend((high[None, :2] - actions[:, :2]).reshape(-1))
        return np.asarray(values, dtype=np.float64)

    sample_clearances = np.asarray(
        [
            minimum_segment_clearance_xy(start, sample, centers, radii)
            for sample in samples
        ]
    )
    sample_progress = np.asarray(
        [closest_target_progress(start, sample, target) for sample in samples]
    )
    order = np.lexsort((-sample_progress, -sample_clearances))
    starts = [nominal_positions[:, :2], model.mean_xy.reshape(-1, 2)]
    for index in order:
        starts.append(samples[index, :, :2])
        if len(starts) >= config.maximum_multistarts:
            break

    constraints = ({"type": "ineq", "fun": constraint_values},)
    candidates = []
    for start_index, initial in enumerate(starts):
        result = minimize(
            objective,
            np.asarray(initial, dtype=np.float64).reshape(-1),
            method="SLSQP",
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
        candidates.append(
            (
                feasible,
                objective(result.x),
                -minimum_value,
                start_index,
                result,
            )
        )

    candidates.sort(
        key=lambda item: (
            not item[0],
            item[1] if item[0] else item[2],
            item[3],
        )
    )
    feasible, selected_objective, _, selected_start, selected = candidates[0]
    corrected_positions, corrected_actions = unpack(selected.x)
    support_d2 = model.support_distance_d2(corrected_positions[:, :2])
    return model, ProjectionResult(
        feasible=feasible,
        corrected_positions=corrected_positions,
        corrected_actions=corrected_actions,
        objective_d2_from_nominal=float(selected_objective),
        support_distance_d2=support_d2,
        support_ratio=float(support_d2 / model.support_threshold_d2),
        baseline_min_clearance_m=nominal_clearance,
        corrected_min_clearance_m=minimum_segment_clearance_xy(
            start,
            corrected_positions,
            centers,
            radii,
        ),
        baseline_target_progress_m=nominal_progress,
        corrected_target_progress_m=closest_target_progress(
            start,
            corrected_positions,
            target,
        ),
        maximum_waypoint_displacement_m=float(
            np.max(np.linalg.norm(corrected_positions - nominal_positions, axis=1))
        ),
        minimum_constraint_value=float(np.min(constraint_values(selected.x))),
        solver_success=bool(selected.success),
        solver_status=int(selected.status),
        solver_message=str(selected.message),
        solver_iterations=int(selected.nit),
        selected_start=int(selected_start),
        starts_attempted=len(starts),
    )
