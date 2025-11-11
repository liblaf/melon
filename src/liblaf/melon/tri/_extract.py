from collections.abc import Iterable
from typing import Any

import pyvista as pv
from pyvista import VectorLike

from liblaf.melon import io

from ._group import select_groups


def extract_cells(
    mesh: Any, ind: int | VectorLike[int], *, invert: bool = False
) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    cells: pv.UnstructuredGrid = mesh.extract_cells(ind, invert=invert)  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = cells.extract_surface()  # pyright: ignore[reportAssignmentType]
    return surface


def extract_groups(
    mesh: Any, groups: int | str | Iterable[int | str], *, invert: bool = False
) -> pv.PolyData:
    return extract_cells(mesh, select_groups(mesh, groups), invert=invert)


def extract_points(
    mesh: Any,
    ind: int | VectorLike[int] | VectorLike[bool],
    *,
    adjacent_cells: bool = True,
    include_cells: bool = True,
) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    points: pv.UnstructuredGrid = mesh.extract_points(
        ind, adjacent_cells=adjacent_cells, include_cells=include_cells
    )  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = points.extract_surface()  # pyright: ignore[reportAssignmentType]
    return surface
