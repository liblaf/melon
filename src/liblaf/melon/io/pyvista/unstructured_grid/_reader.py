from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

load_unstructured_grid: ReaderDispatcher[pv.UnstructuredGrid] = ReaderDispatcher(
    pv.UnstructuredGrid
)
"""Load an unstructured volume mesh with PyVista."""


@load_unstructured_grid.register_fallback
def _read(path: Path, /, **kwargs) -> pv.UnstructuredGrid:
    return pv.read(path, **kwargs)
