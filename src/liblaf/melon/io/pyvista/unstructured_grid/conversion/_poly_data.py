from typing import override

import pyvista as pv

from liblaf.melon.io import abc


class PolyDataToUnstructuredGrid(abc.AbstractConverter):
    type_from: type = pv.PolyData
    type_to: type = pv.UnstructuredGrid

    @override
    def convert(self, obj: pv.PolyData, /, **kwargs) -> pv.UnstructuredGrid:
        return obj.cast_to_unstructured_grid()
