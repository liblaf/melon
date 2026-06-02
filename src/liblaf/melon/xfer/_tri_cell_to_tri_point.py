from collections.abc import Iterable

import pyvista as pv


def tri_cell_to_tri_point(
    source: pv.PolyData,
    target: pv.PolyData,
    names: Iterable[str],
    *,
    tolerance: float | None = None,
    snap_to_closest_point: bool = True,
) -> pv.PolyData:
    src: pv.PolyData = pv.PolyData()
    src.copy_structure(source)
    src.cell_data.update({name: source.cell_data[name] for name in names})
    result: pv.PolyData = target.sample(
        src, tolerance=tolerance, snap_to_closest_point=snap_to_closest_point
    )
    result.point_data.pop("vtkGhostType", None)
    target.point_data.update(result.point_data)
    return target
