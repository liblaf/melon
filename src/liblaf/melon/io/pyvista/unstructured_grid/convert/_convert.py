import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher

from ._data_set import DataSetToUnstructuredGrid

as_unstructured_grid = ConverterDispatcher(pv.UnstructuredGrid)
as_unstructured_grid.register(DataSetToUnstructuredGrid())
