from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .config import EnvironmentConfig
from .env import LayoutSpec, Obstacle, swept_clearance


def _free(
    point: np.ndarray,
    obstacles: tuple[Obstacle, ...],
    config: EnvironmentConfig,
    margin: float,
) -> bool:
    low = config.workspace_low + config.agent_radius + margin
    high = config.workspace_high - config.agent_radius - margin
    if np.any(point < low) or np.any(point > high):
        return False
    return all(
        float(np.linalg.norm(point - obstacle.center))
        > obstacle.radius + config.agent_radius + margin
        for obstacle in obstacles
    )


def _line_free(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: tuple[Obstacle, ...],
    config: EnvironmentConfig,
    margin: float,
) -> bool:
    return (
        _free(start, obstacles, config, margin)
        and _free(end, obstacles, config, margin)
        and swept_clearance(start, end, obstacles, config.agent_radius) > margin
    )


def _shortcut(
    path: list[np.ndarray],
    obstacles: tuple[Obstacle, ...],
    config: EnvironmentConfig,
    margin: float,
) -> list[np.ndarray]:
    result = [path[0]]
    index = 0
    while index < len(path) - 1:
        target = len(path) - 1
        while target > index + 1 and not _line_free(
            path[index], path[target], obstacles, config, margin
        ):
            target -= 1
        result.append(path[target])
        index = target
    return result


def shortest_path(
    layout: LayoutSpec,
    config: EnvironmentConfig | None = None,
    *,
    margin: float = 0.02,
    resolution: float = 0.04,
) -> list[np.ndarray] | None:
    """Deterministic A* path, invariant to obstacle input order."""

    cfg = config or EnvironmentConfig()
    obstacles = tuple(sorted(layout.obstacles))
    start = np.asarray(layout.start, dtype=np.float64)
    goal = np.asarray(layout.goal, dtype=np.float64)
    if _line_free(start, goal, obstacles, cfg, margin):
        return [start, goal]

    low = cfg.workspace_low + cfg.agent_radius + margin
    high = cfg.workspace_high - cfg.agent_radius - margin
    count = int(round((high - low) / resolution)) + 1
    coordinates = np.linspace(low, high, count)

    def nearest(point: np.ndarray) -> tuple[int, int]:
        index = np.rint((point - low) / (high - low) * (count - 1)).astype(int)
        index = np.clip(index, 0, count - 1)
        return int(index[0]), int(index[1])

    start_node, goal_node = nearest(start), nearest(goal)
    nodes = {start_node: start, goal_node: goal}

    def point(node: tuple[int, int]) -> np.ndarray:
        return nodes.get(
            node,
            np.asarray((coordinates[node[0]], coordinates[node[1]])),
        )

    frontier: list[tuple[float, float, int, tuple[int, int]]] = []
    serial = itertools.count()
    heapq.heappush(frontier, (0.0, 0.0, next(serial), start_node))
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start_node: 0.0}
    offsets = tuple(
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx != 0 or dy != 0
    )
    found = False
    while frontier:
        _, current_cost, _, current = heapq.heappop(frontier)
        if current == goal_node:
            found = True
            break
        if current_cost > cost[current] + 1e-12:
            continue
        current_point = point(current)
        for dx, dy in offsets:
            neighbor = current[0] + dx, current[1] + dy
            if not (0 <= neighbor[0] < count and 0 <= neighbor[1] < count):
                continue
            neighbor_point = point(neighbor)
            if not _line_free(current_point, neighbor_point, obstacles, cfg, margin):
                continue
            edge = float(np.linalg.norm(neighbor_point - current_point))
            proposal = current_cost + edge
            if proposal + 1e-12 >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = proposal
            parent[neighbor] = current
            heuristic = float(np.linalg.norm(neighbor_point - goal))
            heapq.heappush(
                frontier,
                (proposal + heuristic, proposal, next(serial), neighbor),
            )
    if not found:
        return None
    node_path = [goal_node]
    while node_path[-1] != start_node:
        node_path.append(parent[node_path[-1]])
    node_path.reverse()
    raw = [point(node).copy() for node in node_path]
    raw[0], raw[-1] = start, goal
    return _shortcut(raw, obstacles, cfg, margin)


def path_to_actions(
    path: list[np.ndarray],
    max_step: float,
    *,
    ease: bool = True,
) -> np.ndarray:
    """Interpolate a waypoint path into clipped deltas with mild easing."""

    actions: list[np.ndarray] = []
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        count = max(1, math.ceil(distance / max_step))
        if ease and count >= 4:
            weights = np.sin(np.linspace(0.2, math.pi - 0.2, count))
            weights = weights / weights.sum()
        else:
            weights = np.full(count, 1.0 / count)
        segment_actions = [delta * weight for weight in weights]
        while max(float(np.linalg.norm(a)) for a in segment_actions) > max_step:
            count += 1
            weights = np.full(count, 1.0 / count)
            segment_actions = [delta * weight for weight in weights]
        actions.extend(segment_actions)
    return np.asarray(actions, dtype=np.float32)


def oracle_actions(
    layout: LayoutSpec,
    config: EnvironmentConfig | None = None,
    *,
    mode: str = "shortest",
) -> np.ndarray:
    cfg = config or EnvironmentConfig()
    if layout.difficulty == 1 and mode in {"upper", "lower"}:
        candidate = d1_mode_path(layout, cfg, mode=mode)
        if candidate is not None:
            return path_to_actions(candidate, cfg.max_step)
    path = shortest_path(layout, cfg, margin=cfg.planner_margin)
    if path is None:
        raise RuntimeError(f"oracle could not solve layout {layout.identifier}")
    return path_to_actions(path, cfg.max_step)


def d1_mode_path(
    layout: LayoutSpec,
    config: EnvironmentConfig | None = None,
    *,
    mode: str,
) -> list[np.ndarray] | None:
    """Return a genuine upper/lower D1 route without shortest-path fallback."""

    if layout.difficulty != 1 or mode not in {"upper", "lower"}:
        raise ValueError("D1 mode must be upper or lower")
    cfg = config or EnvironmentConfig()
    obstacle = layout.obstacles[0]
    sign = 1.0 if mode == "upper" else -1.0
    waypoint = np.asarray(
        (
            obstacle.x,
            obstacle.y
            + sign
            * (obstacle.radius + cfg.agent_radius + cfg.planner_margin + 0.08),
        )
    )
    candidate = [
        np.asarray(layout.start, dtype=np.float64),
        waypoint,
        np.asarray(layout.goal, dtype=np.float64),
    ]
    obstacles = tuple(sorted(layout.obstacles))
    if all(
        _line_free(a, b, obstacles, cfg, cfg.planner_margin)
        for a, b in zip(candidate[:-1], candidate[1:])
    ):
        return candidate
    return None


def rollout_oracle(
    layout: LayoutSpec,
    config: EnvironmentConfig | None = None,
    *,
    mode: str = "shortest",
) -> tuple[bool, bool, int]:
    from .env import PointNavEnv

    cfg = config or EnvironmentConfig()
    env = PointNavEnv(layout=layout, config=cfg)
    env.reset(layout.seed, layout)
    done = False
    for action in oracle_actions(layout, cfg, mode=mode):
        _, _, done, info = env.step(action)
        if done:
            return bool(info["success"]), bool(info["collision"]), env.state.step_count
    while not done:
        delta = np.asarray(layout.goal) - env.state.position
        _, _, done, info = env.step(delta)
    return bool(info["success"]), bool(info["collision"]), env.state.step_count
