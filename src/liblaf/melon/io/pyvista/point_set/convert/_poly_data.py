from typing import Any, override

import pyvista as pv

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractConverter


class PolyDataToPointSet(AbstractConverter):
    @override
    def match_from(self, data: Any) -> bool:
        return utils.is_poly_data(data)

    @override
    def convert(self, data: pv.PolyData, /, **kwargs) -> pv.PointSet:
        return data.cast_to_pointset(**kwargs)
