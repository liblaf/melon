from collections.abc import Collection, Iterable

import pyvista as pv

from liblaf.melon import utils

from ._tet_cell_to_point import transfer_tet_cell_to_point


def transfer_tet_cell(
    source: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    data: str | Iterable[str] | None = None,
    **kwargs,
) -> pv.UnstructuredGrid:
    from liblaf.melon import tet

    data: Collection[str] = utils.as_names(data, source.cell_data)
    output: pv.UnstructuredGrid = transfer_tet_cell_to_point(
        source, target, data, **kwargs
    )
    output = tet.point_data_to_cell_data(output, data)
    for name in data:
        target.cell_data[name] = output.cell_data[name]
    return output
