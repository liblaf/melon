import os
from pathlib import Path

from liblaf import grapes
from liblaf.melon.io._const import SUFFIXES


def get_landmarks_path(path: str | os.PathLike[str]) -> Path:
    path: Path = grapes.as_path(path)
    if path.suffix in SUFFIXES:
        return path.with_suffix(".landmarks.json")
    return path
