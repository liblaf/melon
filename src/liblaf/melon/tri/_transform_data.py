from collections.abc import Collection, Iterable

import pyvista as pv

from liblaf.melon import io, utils


def cell_data_to_point_data(
    mesh: pv.PolyData, data: str | Iterable[str] | None = None
) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    data: Collection[str] = utils.as_names(data, mesh.cell_data)
    source: pv.PolyData = pv.PolyData()
    source.copy_structure(mesh)
    source.cell_data.update({name: mesh.cell_data[name] for name in data}, copy=False)
    output: pv.PolyData = source.cell_data_to_point_data()  # pyright: ignore[reportAssignmentType]
    for name in data:
        mesh.point_data[name] = output.point_data[name]
    return mesh


def point_data_to_cell_data(
    mesh: pv.PolyData, data: str | Iterable[str] | None = None
) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    data: Collection[str] = utils.as_names(data, mesh.point_data)
    source: pv.PolyData = pv.PolyData()
    source.copy_structure(mesh)
    source.point_data.update({name: mesh.point_data[name] for name in data}, copy=False)
    output: pv.PolyData = source.point_data_to_cell_data()  # pyright: ignore[reportAssignmentType]
    for name in data:
        mesh.cell_data[name] = output.cell_data[name]
    return mesh
