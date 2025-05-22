from typing import Any, override

import pyvista as pv
import trimesh as tm

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractConverter


class TrimeshToPointSet(AbstractConverter):
    @override
    def match_from(self, data: Any, /) -> bool:
        return utils.is_trimesh(data)

    @override
    def convert(self, data: tm.Trimesh, /, **kwargs) -> pv.PointSet:
        point_normals: bool = kwargs.pop("point_normals", False)
        result: pv.PointSet = pv.PointSet(data.vertices)
        if point_normals:
            result.point_data["Normals"] = data.vertex_normals
        return result
