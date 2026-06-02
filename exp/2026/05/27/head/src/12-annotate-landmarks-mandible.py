from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    left: Path = cherries.input("00-sculptor-mandible.ply")
    right: Path = cherries.input("11-mandible.ply")


def main(config: Config) -> None:
    left: pv.PolyData = melon.io.load_polydata(config.left)
    right: pv.PolyData = melon.io.load_polydata(config.right)
    left_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(config.left)
    right_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(config.right)
    left_landmarks, right_landmarks = melon.ext.annotate_landmarks(
        left, right, left_landmarks=left_landmarks, right_landmarks=right_landmarks
    )
    melon.io.save_landmarks(left_landmarks, config.left)
    melon.io.save_landmarks(right_landmarks, config.right)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
