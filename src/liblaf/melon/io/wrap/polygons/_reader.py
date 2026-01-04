from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Integer

from liblaf import grapes

from ._utils import get_polygons_path

if TYPE_CHECKING:
    from _typeshed import StrPath


def load_polygons(path: StrPath) -> Integer[np.ndarray, " N"]:
    path: Path = get_polygons_path(path)
    data: list[int] = grapes.load(path)
    return np.asarray(data)
