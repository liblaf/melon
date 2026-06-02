import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    left: Path = cherries.input("00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj")
    right: Path = cherries.input("20-skin-smoothed.ply")
    left_landmarks: Path = cherries.output(
        "00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.skin.landmarks.json"
    )


def main(cfg: Config) -> None:
    left: pv.PolyData = melon.io.load_polydata(cfg.left)
    right: pv.PolyData = melon.io.load_polydata(cfg.right)
    left_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(
        cfg.left_landmarks
    )
    right_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(cfg.right)
    left_landmarks, right_landmarks = melon.ext.annotate_landmarks(
        left, right, left_landmarks=left_landmarks, right_landmarks=right_landmarks
    )
    melon.io.save_landmarks(left_landmarks, cfg.left_landmarks)
    melon.io.save_landmarks(right_landmarks, cfg.right)


if __name__ == "__main__":
    cherries.main(main)
