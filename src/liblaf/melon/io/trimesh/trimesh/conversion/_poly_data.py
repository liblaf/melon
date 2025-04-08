from typing import override

import pyvista as pv
import trimesh as tm

from liblaf.melon.io import abc


class PolyDataToTrimesh(abc.AbstractConverter):
    type_from: type = pv.PolyData
    type_to: type = tm.Trimesh

    @override
    def convert(self, obj: pv.PolyData, /, **kwargs) -> tm.Trimesh:
        obj = obj.triangulate()  # pyright: ignore[reportAssignmentType]
        return tm.Trimesh(obj.points, obj.regular_faces)
