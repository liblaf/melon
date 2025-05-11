from collections.abc import Container
from pathlib import Path
from typing import Any, override

from liblaf.melon.io.abc import AbstractWriter
from liblaf.melon.io.pyvista.unstructured_grid.convert import as_unstructured_grid
from liblaf.melon.typed import PathLike


class UnstructuredGridWriter(AbstractWriter):
    extensions: Container[str] = {".vtu"}

    @override
    def save(self, path: PathLike, data: Any, /, **kwargs) -> None:
        import pyvista as pv

        path = Path(path)
        data: pv.UnstructuredGrid = as_unstructured_grid(data)
        return data.save(path, **kwargs)
