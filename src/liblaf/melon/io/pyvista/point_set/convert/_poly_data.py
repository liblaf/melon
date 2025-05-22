from typing import Any, override

import pyvista as pv

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractConverter


class PolyDataToPointSet(AbstractConverter):
    @override
    def match_from(self, data: Any, /) -> bool:
        return utils.is_poly_data(data)

    @override
    def convert(self, data: pv.PolyData, /, **kwargs) -> pv.PointSet:
        point_normals: bool = kwargs.pop("point_normals", False)
        data.active_scalars_name = None
        result: pv.PointSet = data.cast_to_pointset(**kwargs)
        if point_normals:
            result.point_data["Normals"] = data.point_normals
        return result
