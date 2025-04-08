import os
from pathlib import Path
from typing import override

import pyvista as pv

from liblaf import grapes
from liblaf.melon.io import abc


def load_image_data(path: str | os.PathLike[str]) -> pv.ImageData:
    path = Path(path)
    if path.is_file() and path.name == "DIRFILE":
        path = path.parent
    if path.is_dir() and (path / "DIRFILE").exists():
        return pv.read(path, force_ext=".dcm")  # pyright: ignore[reportReturnType]
    return pv.read(path)  # pyright: ignore[reportReturnType]


class ImageDataReader(abc.AbstractReader):
    precedence: int = -1  # discourage this reader, use `DICOMReader` instead

    @override
    def match_path(self, path: str | os.PathLike[str]) -> bool:
        path: Path = grapes.as_path(path)
        if path.is_file() and path.name == "DIRFILE":
            return True
        if path.is_dir() and (path / "DIRFILE").exists():
            return True
        return path.suffix in {".dcm", ".vti"}

    @override
    def load(self, path: str | os.PathLike[str], /, **kwargs) -> pv.ImageData:
        return load_image_data(path)
