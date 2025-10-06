from typing import Any, cast

import pyvista as pv

from liblaf.melon import io


def merge_points(mesh: Any, tolerance: float = 0.0) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    result: pv.PolyData = cast("pv.PolyData", mesh.merge_points(tolerance=tolerance))
    result.field_data.update(mesh.field_data)
    return result
