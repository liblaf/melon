from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf.melon import io, plugin


def main(
    left_file: Path,
    right_file: Path,
    *,
    left_landmarks_file: Path | None = None,
    right_landmarks_file: Path | None = None,
) -> None:
    left: pv.PolyData = io.load_poly_data(left_file)
    right: pv.PolyData = io.load_poly_data(right_file)
    if left_landmarks_file is None:
        left_landmarks_file = io.get_landmarks_path(left_file)
    if right_landmarks_file is None:
        right_landmarks_file = io.get_landmarks_path(right_file)
    left_landmarks: Float[np.ndarray, "L 3"] | None = (
        io.load_landmarks(left_landmarks_file) if left_landmarks_file.exists() else None
    )
    right_landmarks: Float[np.ndarray, "L 3"] | None = (
        io.load_landmarks(right_landmarks_file)
        if right_landmarks_file.exists()
        else None
    )
    left_landmarks, right_landmarks = plugin.annotate_landmarks(
        left, right, left_landmarks=left_landmarks, right_landmarks=right_landmarks
    )
    io.save_landmarks(left_landmarks_file, left_landmarks)
    io.save_landmarks(right_landmarks_file, right_landmarks)
