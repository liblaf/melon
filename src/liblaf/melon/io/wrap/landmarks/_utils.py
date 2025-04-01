from pathlib import Path

from liblaf import melon
from liblaf.melon.typed import PathLike


def get_landmarks_path(path: PathLike) -> Path:
    path = Path(path)
    if path.suffix in melon.io.SUFFIXES:
        return path.with_suffix(".landmarks.json")
    return path
