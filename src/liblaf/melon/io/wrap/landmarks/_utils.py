import os
from pathlib import Path

from liblaf import grapes


def get_landmarks_path(path: str | os.PathLike[str]) -> Path:
    path: Path = grapes.as_path(path)
    if path.suffix != ".json":
        return path.with_suffix(".landmarks.json")
    return path
