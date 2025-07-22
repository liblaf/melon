from collections.abc import Iterable
from typing import Any

import pyvista as pv
from jaxtyping import Bool, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._selection import select_groups


def extract_cells(
    mesh: Any,
    ind: Bool[ArrayLike, " C"] | Integer[ArrayLike, " S"],
    *,
    invert: bool = False,
) -> pv.PolyData:
    mesh: pv.PolyData = io.as_poly_data(mesh)
    cells: pv.UnstructuredGrid = mesh.extract_cells(ind, invert=invert)  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = cells.extract_surface()  # pyright: ignore[reportAssignmentType]
    return surface


def extract_groups(
    mesh: Any, groups: int | str | Iterable[int | str], *, invert: bool = False
) -> pv.PolyData:
    return extract_cells(mesh, select_groups(mesh, groups), invert=invert)


def extract_points(
    mesh: Any,
    ind: Bool[ArrayLike, " N"] | Integer[ArrayLike, " S"],
    *,
    adjacent_cells: bool = True,
    include_cells: bool = True,
) -> pv.PolyData:
    mesh: pv.PolyData = io.as_poly_data(mesh)
    points: pv.UnstructuredGrid = mesh.extract_points(
        ind, adjacent_cells=adjacent_cells, include_cells=include_cells
    )  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = points.extract_surface()  # pyright: ignore[reportAssignmentType]
    return surface
