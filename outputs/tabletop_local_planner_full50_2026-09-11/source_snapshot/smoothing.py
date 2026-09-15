"""Collision-checked shortening and cubic Bezier corner rounding.

The original 10 mm clearance is preserved. Curves are recursively flattened;
a midpoint separating-plane bound additionally covers each control hull's
maximum deviation from its chord. Runtime checks the actual commanded chords.
"""

import numpy as np


def path_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def split_cubic(control):
    a = (control[:-1] + control[1:]) / 2
    b = (a[:-1] + a[1:]) / 2
    c = (b[0] + b[1]) / 2
    return np.array([control[0], a[0], b[0], c]), np.array([c, b[1], a[2], control[3]])


def flatten_cubic(control, tolerance=0.0005, depth=0):
    """Return chords and conservative deviation bounds for their curve pieces."""
    d = control[-1] - control[0]
    denom = np.dot(d, d)
    t = np.clip((control - control[0]) @ d / max(denom, 1e-20), 0, 1)
    error = float(np.linalg.norm(control - (control[0] + t[:, None] * d), axis=1).max())
    if error <= tolerance:
        return [(control[0], control[-1], error)]
    if depth >= 20:
        raise ValueError("Cubic flattening did not converge")
    left, right = split_cubic(control)
    return flatten_cubic(left, tolerance, depth + 1) + flatten_cubic(
        right, tolerance, depth + 1
    )


def compact_bezier(geometry, route, rotation):
    original = route["points"].copy()
    points = original.copy()
    # Keep 5 mm beyond the unchanged certificate margin for corner rounding.
    for _ in range(12):
        for i in range(1, len(points) - 1):
            candidate = points[i] + 0.15 * (
                (points[i - 1] + points[i + 1]) / 2 - points[i]
            )
            if (
                geometry.segment_clearance(points[i - 1], rotation, candidate, rotation)
                >= 0.005
                and geometry.segment_clearance(
                    candidate, rotation, points[i + 1], rotation
                )
                >= 0.005
            ):
                points[i] = candidate
    output = [points[0]]
    curves = []
    min_curve_bound = np.inf
    for i in range(1, len(points) - 1):
        incoming = points[i] - points[i - 1]
        outgoing = points[i + 1] - points[i]
        before, after = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        accepted = False
        if min(before, after) > 1e-8:
            incoming /= before
            outgoing /= after
            cut = 0.45 * min(before, after)
            for _ in range(9):
                a, b = points[i] - cut * incoming, points[i] + cut * outgoing
                control = np.array(
                    [a, a + 0.55 * cut * incoming, b - 0.55 * cut * outgoing, b]
                )
                chords = flatten_cubic(control)
                bound = min(
                    geometry.segment_clearance(x, rotation, y, rotation) - error
                    for x, y, error in chords
                )
                if (
                    bound >= 0
                    and geometry.segment_clearance(output[-1], rotation, a, rotation)
                    >= 0
                ):
                    if np.linalg.norm(output[-1] - a) > 1e-10:
                        output.append(a)
                    output.extend(y for _, y, _ in chords)
                    curves.append(control)
                    min_curve_bound = min(min_curve_bound, bound)
                    accepted = True
                    break
                cut *= 0.5
        if not accepted:
            output.append(points[i])
    output.append(points[-1])
    output = np.array(output)
    # Preserve endpoints and validate joins too. A safe polyline remains fallback.
    if any(
        geometry.segment_clearance(a, rotation, b, rotation) < 0
        for a, b in zip(output[:-1], output[1:])
    ):
        output, curves = original, []
        min_curve_bound = np.inf
    return {
        **route,
        "points": output,
        "original_points": original,
        "compact_points": points,
        "bezier_controls": curves,
        "original_length": path_length(original),
        "length": path_length(output),
        "curve_clearance_lower_m": float(min_curve_bound) if curves else None,
        "style": "compact_bezier",
    }


def advance_path(points, progress, position, horizon, increment=0.006):
    """Monotone observed progress and arc-length targets across rounded corners.

    Project only up to 7 cm ahead of the previous observed progress to avoid
    jumping to a nearby later path leg. Sampling is by length, not Bezier t.
    """
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    candidates = (
        [(np.linalg.norm(position - points[0]), progress)] if progress == 0 else []
    )
    for i, length in enumerate(lengths):
        if (
            length < 1e-12
            or cumulative[i + 1] < progress
            or cumulative[i] > progress + 0.07
        ):
            continue
        delta = points[i + 1] - points[i]
        low = max(0.0, (progress - cumulative[i]) / length)
        high = min(1.0, (progress + 0.07 - cumulative[i]) / length)
        t = np.clip(np.dot(position - points[i], delta) / length**2, low, high)
        candidates.append(
            (
                np.linalg.norm(position - (points[i] + t * delta)),
                cumulative[i] + t * length,
            )
        )
    if candidates:
        progress = max(progress, min(candidates)[1])
    distances = np.minimum(
        cumulative[-1], progress + increment * np.arange(1, horizon + 1)
    )
    targets = np.column_stack(
        [np.interp(distances, cumulative, points[:, axis]) for axis in range(3)]
    )
    return progress, targets, cumulative[-1]
