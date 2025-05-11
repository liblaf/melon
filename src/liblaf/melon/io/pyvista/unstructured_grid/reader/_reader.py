import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

from ._pyvista import UnstructuredGridReader

load_unstructured_grid = ReaderDispatcher(pv.UnstructuredGrid)
load_unstructured_grid.register(UnstructuredGridReader())
