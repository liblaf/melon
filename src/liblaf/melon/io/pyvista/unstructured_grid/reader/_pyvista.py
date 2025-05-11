from collections.abc import Container
from pathlib import Path
from typing import override

import pyvista as pv

from liblaf.melon.io.abc import AbstractReader
from liblaf.melon.typed import PathLike


class UnstructuredGridReader(AbstractReader):
    extensions: Container[str] = {".msh", ".vtk", ".vtu"}

    @override
    def load(self, path: PathLike, /, **kwargs) -> pv.UnstructuredGrid:
        return pv.read(Path(path), **kwargs)
