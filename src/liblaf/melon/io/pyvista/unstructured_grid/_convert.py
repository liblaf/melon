import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher

as_unstructured_grid: ConverterDispatcher[pv.UnstructuredGrid] = ConverterDispatcher(
    pv.UnstructuredGrid
)


@as_unstructured_grid.register(pv.DataSet)
def pyvista_to_unstructured_grid(obj: pv.DataSet, **kwargs) -> pv.UnstructuredGrid:
    return obj.cast_to_unstructured_grid(**kwargs)
