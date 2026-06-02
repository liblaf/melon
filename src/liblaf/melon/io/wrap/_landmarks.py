from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from _typeshed import StrPath


def load_landmarks(path: StrPath) -> Float[np.ndarray, "L 3"]:
    path: Path = _infer_path(path)
    if not path.exists():
        return np.zeros((0, 3))
    with path.open() as fp:
        data: list[dict[str, float]] = json.load(fp)
    return np.asarray([[point["x"], point["y"], point["z"]] for point in data])


def save_landmarks(landmarks: Float[ArrayLike, "L 3"], path: StrPath) -> None:
    landmarks: Float[np.ndarray, "L 3"] = np.asarray(landmarks)
    path: Path = _infer_path(path)
    data: list[dict[str, float]] = [{"x": x, "y": y, "z": z} for x, y, z in landmarks]
    with path.open("w") as fp:
        json.dump(data, fp)


def _infer_path(path: StrPath) -> Path:
    path: Path = Path(path)
    if path.suffix != ".json":
        return path.with_suffix(".landmarks.json")
    return path
