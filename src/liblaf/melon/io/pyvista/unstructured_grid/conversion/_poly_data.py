import pyvista as pv

from liblaf import melon


class PolyDataToUnstructuredGrid(melon.io.AbstractConverter):
    type_from = pv.PolyData
    type_to = pv.UnstructuredGrid

    def convert(self, obj: pv.PolyData) -> pv.UnstructuredGrid:
        return obj.cast_to_unstructured_grid()
