from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

load_unstructured_grid: ReaderDispatcher[pv.UnstructuredGrid] = ReaderDispatcher(
    pv.UnstructuredGrid
)


@load_unstructured_grid.register(".msh", ".vtu")
def load_unstructured_grid_pyvista(path: Path, **kwargs) -> pv.UnstructuredGrid:
    return pv.read(path, **kwargs)  # pyright: ignore[reportReturnType]
