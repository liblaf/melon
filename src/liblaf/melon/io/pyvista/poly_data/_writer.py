import os
from collections.abc import Container
from pathlib import Path
from typing import Any, override

import pyvista as pv

from liblaf import grapes
from liblaf.melon.io import abc

from ._save_obj import save_obj
from .conversion import as_poly_data


class PolyDataWriter(abc.AbstractWriter):
    # exclude `.vtk` because it is ambiguous
    extensions: Container[str] = {".geo", ".iv", ".obj", ".ply", ".stl", ".vtp"}

    @override
    def save(self, path: str | os.PathLike[str], obj: Any, /, **kwargs) -> None:
        path: Path = grapes.as_path(path)
        obj: pv.PolyData = as_poly_data(obj)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".obj":
            save_obj(path, obj)
        else:
            obj.save(path)
