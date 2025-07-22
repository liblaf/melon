from typing import Any, override

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractConverter


class PolyDataToTrimesh(AbstractConverter):
    @override
    def match_from(self, data: Any) -> bool:
        return utils.is_poly_data(data)

    @override
    def convert(self, data: Any, /, **kwargs) -> Any:
        import pyvista as pv
        import trimesh as tm

        assert isinstance(data, pv.PolyData)
        data = data.triangulate()
        return tm.Trimesh(data.points, data.regular_faces, **kwargs)
