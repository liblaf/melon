from typing import Any

import pyvista as pv

from liblaf.melon import io


def compute_area(mesh: Any) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    result: pv.PolyData = mesh.compute_cell_sizes(length=False, area=True, volume=False)  # pyright: ignore[reportAssignmentType]
    return result
