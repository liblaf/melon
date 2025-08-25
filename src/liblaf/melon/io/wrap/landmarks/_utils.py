import os
from pathlib import Path


def get_landmarks_path(path: str | os.PathLike[str]) -> Path:
    path: Path = Path(path)
    if path.suffix != ".json":
        return path.with_suffix(".landmarks.json")
    return path
