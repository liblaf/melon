import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher

as_structured_grid: ConverterDispatcher[pv.StructuredGrid] = ConverterDispatcher(
    pv.StructuredGrid
)
