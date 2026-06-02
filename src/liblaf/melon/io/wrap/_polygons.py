from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Integer
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from _typeshed import StrPath


def save_polygons(
    polygons: Bool[ArrayLike, " full"] | Integer[ArrayLike, " selection"], path: StrPath
) -> None:
    path: Path = Path(path)
    polygons: Bool[np.ndarray, " full"] | Integer[np.ndarray, " selection"] = (
        np.asarray(polygons)
    )
    if np.isdtype(polygons.dtype, "bool"):
        polygons: Integer[np.ndarray, " selection"] = np.flatnonzero(polygons)
    with path.open("w") as fp:
        json.dump(polygons.tolist(), fp)
