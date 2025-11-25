from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = cherries.input("00-sculptor-cranium.ply")
    target: Path = cherries.input("13-cranium.vtp")
    output: Path = cherries.output("14-cranium.ply")


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_polydata(cfg.source)
    target: pv.PolyData = melon.load_polydata(cfg.target)
    source_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.source)
    target_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.target)
    result: pv.PolyData = melon.tri.fast_wrapping(
        source,
        target,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
    )

    melon.save(cfg.output, result)
    melon.save_landmarks(cfg.output, target_landmarks)


if __name__ == "__main__":
    cherries.main(main)
