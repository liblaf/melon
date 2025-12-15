from collections.abc import Collection, Iterable

import pyvista as pv

from liblaf.melon import utils


def transfer_tet_cell_to_point(
    source: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    data: str | Iterable[str] | None = None,
    *,
    categorical: bool = False,
    snap_to_closest_point: bool = True,
    tolerance: float | None = 1e-6,
    **kwargs,
) -> pv.UnstructuredGrid:
    data: Collection[str] = utils.as_names(data, source.cell_data)
    source_filtered: pv.UnstructuredGrid = pv.UnstructuredGrid()
    source_filtered.copy_structure(source)
    source_filtered.cell_data.update(
        {name: source.cell_data[name] for name in data}, copy=False
    )
    output: pv.UnstructuredGrid = target.sample(
        source_filtered,
        categorical=categorical,
        snap_to_closest_point=snap_to_closest_point,
        tolerance=tolerance,
        **kwargs,
    )  # pyright: ignore[reportAssignmentType]
    for name in data:
        target.point_data[name] = output.point_data[name]
    return target
