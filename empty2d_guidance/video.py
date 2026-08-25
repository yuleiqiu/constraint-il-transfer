"""Offline paired-video rendering from recorded Empty2D rollout rows."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .environment import EnvEmpty2DExtraSphere, EnvEmpty2DExtraSquare


def _chunk_at_step(row: dict, step: int) -> dict | None:
    selected = None
    for chunk in row.get("chunks", []):
        if chunk["start_step"] <= step:
            selected = chunk
        else:
            break
    return selected


def _frame(row: dict, step: int) -> np.ndarray:
    environment_kwargs = {
        "horizon": row.get("horizon", 320),
        "obstacle_center": row.get("obstacle_center", (0.0, 0.0)),
    }
    environment = (
        EnvEmpty2DExtraSphere(**environment_kwargs)
        if row["environment"] == "circle"
        else EnvEmpty2DExtraSquare(**environment_kwargs)
    )
    trajectory = np.asarray(row["trajectory"])
    environment.trajectory = [point.copy() for point in trajectory[: step + 1]]
    environment.position = trajectory[min(step, len(trajectory) - 1)].copy()
    chunk = _chunk_at_step(row, step)
    predicted = None if chunk is None else chunk.get("predicted_chunk")
    closest = None if chunk is None else chunk.get("closest_obstacle_point")
    return environment.render(
        predicted_chunk=predicted,
        closest_obstacle_point=closest,
        safety_distance=row["safety_distance"],
        title=row.get("visualization_label", row.get("condition")),
        size=420,
    )


def render_pair(path: Path, baseline: dict, guided: dict, *, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = max(len(baseline["trajectory"]), len(guided["trajectory"]))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for step in range(length):
            before = _frame(baseline, min(step, len(baseline["trajectory"]) - 1))
            after = _frame(guided, min(step, len(guided["trajectory"]) - 1))
            writer.append_data(np.concatenate((before, after), axis=1))
