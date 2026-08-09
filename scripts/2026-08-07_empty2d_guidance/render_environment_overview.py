"""Render a three-panel overview of the fixed Empty2D environments."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from empty2d_guidance.environment import (
    GOAL,
    EnvEmpty2D,
    EnvEmpty2DExtraSphere,
    EnvEmpty2DExtraSquare,
)


ENVIRONMENTS = (
    ("D0 / Empty", EnvEmpty2D),
    ("Circle / radius 0.3", EnvEmpty2DExtraSphere),
    ("Square / side 0.6", EnvEmpty2DExtraSquare),
)


def label(image: np.ndarray, title: str, environment: EnvEmpty2D) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    status = "COLLISION" if environment.collision_any else "collision-free"
    draw.rectangle((0, 0, canvas.width, 42), fill=(255, 255, 255))
    draw.text((12, 7), f"{title} | {status}", fill=(170, 0, 0) if environment.collision_any else (0, 90, 0))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/empty2d_guidance/environment_overview.mp4"),
    )
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    environments = [(title, constructor()) for title, constructor in ENVIRONMENTS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8) as writer:
        while True:
            panels = [label(env.render(size=400), title, env) for title, env in environments]
            writer.append_data(np.concatenate(panels, axis=1))
            if all(env.is_success() for _, env in environments):
                break
            for _, environment in environments:
                if not environment.is_success():
                    environment.step(GOAL)
        for _ in range(args.fps):
            writer.append_data(np.concatenate(panels, axis=1))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
