from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    floating: Path = cherries.input("00-sculptor-mandible.ply")
    fixed: Path = cherries.input("11-mandible.ply")
    output: Path = cherries.output("13-mandible.ply")


def main(cfg: Config) -> None:
    floating: pv.PolyData = melon.io.load_polydata(cfg.floating)
    fixed: pv.PolyData = melon.io.load_polydata(cfg.fixed)
    floating_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(cfg.floating)
    fixed_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(cfg.fixed)
    result: pv.PolyData = melon.ext.fast_wrapping(
        floating,
        fixed,
        floating_landmarks=floating_landmarks,
        fixed_landmarks=fixed_landmarks,
    )
    melon.save(result, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
