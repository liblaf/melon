import os
from pathlib import Path

import numpy as np
from jaxtyping import Float

from liblaf import grapes

from ._utils import get_landmarks_path


def load_landmarks(path: str | os.PathLike[str]) -> Float[np.ndarray, "N 3"]:
    path: Path = get_landmarks_path(path)
    if not path.exists():
        return np.empty((0, 3), dtype=float)
    data: list[dict[str, float]] = grapes.load(path)
    return np.asarray([[p["x"], p["y"], p["z"]] for p in data])
