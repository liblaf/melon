import os
from collections.abc import Container
from pathlib import Path
from typing import Any, override

import pyvista as pv

from liblaf import grapes
from liblaf.melon.io import abc

from .conversion import as_unstructured_grid


class UnstructuredGridWriter(abc.AbstractWriter):
    # exclude `.vtk` because it is ambiguous
    extensions: Container[str] = {".vtu"}

    @override
    def save(self, path: str | os.PathLike[str], obj: Any, /, **kwargs) -> None:
        path: Path = grapes.as_path(path)
        obj: pv.UnstructuredGrid = as_unstructured_grid(obj)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.save(path)
