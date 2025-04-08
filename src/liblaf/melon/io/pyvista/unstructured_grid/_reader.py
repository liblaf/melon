import os
from collections.abc import Container
from pathlib import Path
from typing import override

import pyvista as pv

from liblaf import grapes
from liblaf.melon.io import abc


def load_unstructured_grid(path: str | os.PathLike[str]) -> pv.UnstructuredGrid:
    path: Path = grapes.as_path(path)
    return pv.read(path)  # pyright: ignore[reportReturnType]


class UnstructuredGridReader(abc.AbstractReader):
    extensions: Container[str] = {".vtu"}

    @override
    def load(self, path: str | os.PathLike[str], /, **kwargs) -> pv.UnstructuredGrid:
        return load_unstructured_grid(path)
