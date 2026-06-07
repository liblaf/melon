from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Integer
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from _typeshed import StrPath


def load_polygons(path: StrPath) -> Integer[np.ndarray, " N"]:
    """Load selected polygon indices from JSON.

    Args:
        path: JSON file containing polygon indices.

    Returns:
        One-dimensional `int32` array of selected polygon indices. Missing files
        return an empty array.
    """
    path: Path = Path(path)
    if not path.exists():
        return np.empty((0,), np.int32)
    with path.open() as fp:
        data: list[int] = json.load(fp)
    polygons: Integer[np.ndarray, " N"] = np.asarray(data, np.int32)
    return polygons


def save_polygons(
    polygons: Bool[ArrayLike, " full"] | Integer[ArrayLike, " selection"], path: StrPath
) -> None:
    """Save selected polygon indices for Wrap projects.

    Boolean masks are converted to their selected indices before serialization.

    Args:
        polygons: Boolean mask over all polygons or explicit polygon indices.
        path: JSON file to write.
    """
    path: Path = Path(path)
    polygons: Bool[np.ndarray, " full"] | Integer[np.ndarray, " selection"] = (
        np.asarray(polygons)
    )
    if np.isdtype(polygons.dtype, "bool"):
        polygons: Integer[np.ndarray, " selection"] = np.flatnonzero(polygons)
    with path.open("w") as fp:
        json.dump(polygons.tolist(), fp)
