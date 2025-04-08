import os
from collections.abc import Container
from pathlib import Path
from typing import override

import pyvista as pv

from liblaf import grapes
from liblaf.melon.io import abc

from ._load_obj import load_obj


def load_poly_data(path: str | os.PathLike[str], /) -> pv.PolyData:
    path: Path = grapes.as_path(path)
    if path.suffix == ".obj":
        return load_obj(path)
    return pv.read(path)  # pyright: ignore[reportReturnType]


class PolyDataReader(abc.AbstractReader):
    extensions: Container[str] = {".obj", ".stl", ".vtp", ".ply"}

    @override
    def load(self, path: str | os.PathLike[str], /, **kwargs) -> pv.PolyData:
        return load_poly_data(path, **kwargs)
