from typing import override

import pyvista as pv

from liblaf.melon.io import abc


class PolyDataToPointSet(abc.AbstractConverter):
    type_from: type = pv.PolyData
    type_to: type = pv.PointSet

    @override
    def convert(self, obj: pv.PolyData, /, **kwargs) -> pv.PointSet:
        return obj.cast_to_pointset()
