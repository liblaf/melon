from collections.abc import Iterable

import pyvista as pv


def tri_cell_to_tri_point_categorical(
    source: pv.PolyData,
    target: pv.PolyData,
    names: Iterable[str],
    *,
    snap_to_closest_point: bool = True,
) -> pv.PolyData:
    src: pv.PolyData = pv.PolyData()
    src.copy_structure(source)
    src.cell_data.update({name: target.cell_data[name] for name in names})
    result: pv.PolyData = target.sample(
        src, categorical=True, snap_to_closest_point=snap_to_closest_point
    )
    target.point_data.update(result.point_data)
    return target
