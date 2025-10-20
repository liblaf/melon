from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

load_structured_grid: ReaderDispatcher[pv.StructuredGrid] = ReaderDispatcher(
    pv.StructuredGrid
)


@load_structured_grid.register(".vts")
def load_structured_grid_pyvista(path: Path, **kwargs) -> pv.StructuredGrid:
    return pv.read(path, **kwargs)  # pyright: ignore[reportReturnType]
