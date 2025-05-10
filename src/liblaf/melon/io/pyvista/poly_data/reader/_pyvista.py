from collections.abc import Container
from pathlib import Path
from typing import override

import pyvista as pv

from liblaf.melon.io.abc import AbstractReader
from liblaf.melon.typed import PathLike


class PolyDataReader(AbstractReader):
    extensions: Container[str] = {".ply", ".stl", ".vtk", ".vtp", ".obj"}

    @override
    def load(self, path: PathLike, /, **kwargs) -> pv.PolyData:
        return pv.read(Path(path), **kwargs)
