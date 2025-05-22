import os
from pathlib import Path

import numpy as np
from jaxtyping import Integer
from numpy.typing import ArrayLike

from liblaf import grapes

from ._utils import get_polygons_path


def save_polygons(
    path: str | os.PathLike[str], polygons: Integer[ArrayLike, " N"]
) -> None:
    path: Path = get_polygons_path(path)
    polygons = np.asarray(polygons)
    grapes.save(path, polygons.tolist())
