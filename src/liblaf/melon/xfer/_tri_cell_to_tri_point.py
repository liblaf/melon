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
    """Sample triangle cell arrays onto target surface points.

    Args:
        source: Surface carrying the source cell-data arrays.
        target: Surface whose points receive sampled point-data arrays.
        names: Cell-data array names to transfer.
        tolerance: Optional PyVista sampling tolerance.
        snap_to_closest_point: Forwarded to [`pyvista.DataSetFilters.sample`][].

    Returns:
        The mutated `target` mesh.
    """
    src: pv.PolyData = pv.PolyData()
    src.copy_structure(source)
    src.cell_data.update({name: source.cell_data[name] for name in names})
    result: pv.PolyData = target.sample(
        src, tolerance=tolerance, snap_to_closest_point=snap_to_closest_point
    )
    result.point_data.pop("vtkGhostType", None)
    target.point_data.update(result.point_data)
    return target
