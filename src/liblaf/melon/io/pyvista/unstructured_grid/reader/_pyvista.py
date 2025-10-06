from collections.abc import Container
from pathlib import Path
from typing import override

import pyvista as pv

from liblaf.melon.io.abc import Reader
from liblaf.melon.typing import PathLike


class UnstructuredGridReader(Reader):
    extensions: Container[str] = {".msh", ".vtk", ".vtu"}

    @override
    def load(self, path: PathLike, /, **kwargs) -> pv.UnstructuredGrid:
        return pv.read(Path(path), **kwargs)  # pyright: ignore[reportReturnType]
