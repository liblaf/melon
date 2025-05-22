from pathlib import Path

import numpy as np
from jaxtyping import Integer

import liblaf.grapes as grapes  # noqa: PLR0402
from liblaf.melon.typed import PathLike

from ._utils import get_polygons_path


def load_polygons(path: PathLike) -> Integer[np.ndarray, " N"]:
    path: Path = get_polygons_path(path)
    data: list[int] = grapes.load(path)
    return np.asarray(data)
