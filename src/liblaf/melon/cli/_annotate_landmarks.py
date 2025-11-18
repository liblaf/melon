from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
from cyclopts import Parameter
from jaxtyping import Float

from liblaf.melon import ext, io


def annotate_landmarks(
    left_path: Annotated[Path, Parameter("left")],
    right_path: Annotated[Path, Parameter("right")],
    /,
    *,
    left_landmarks_path: Annotated[Path | None, Parameter("left-landmarks")] = None,
    right_landmarks_path: Annotated[Path | None, Parameter("right-landmarks")] = None,
) -> None:
    if left_landmarks_path is None:
        left_landmarks_path = left_path
    if right_landmarks_path is None:
        right_landmarks_path = right_path
    left: pv.PolyData = io.load_polydata(left_path)
    left_landmarks: Float[np.ndarray, "landmarks 3"] = io.load_landmarks(
        left_landmarks_path
    )
    right: pv.PolyData = io.load_polydata(right_path)
    right_landmarks: Float[np.ndarray, "landmarks 3"] = io.load_landmarks(
        right_landmarks_path
    )
    left_landmarks, right_landmarks = ext.annotate_landmarks(
        left, right, left_landmarks=left_landmarks, right_landmarks=right_landmarks
    )
    io.save_landmarks(left_landmarks_path, left_landmarks)
    io.save_landmarks(right_landmarks_path, right_landmarks)
