import functools
from typing import Any

import pyvista as pv
from jaxtyping import Bool, Integer
from numpy.typing import ArrayLike


@functools.singledispatch
def extract_cells(
    mesh: Any, selection: Bool[ArrayLike, " C"] | Integer[ArrayLike, " N"]
) -> Any:
    raise NotImplementedError


@extract_cells.register
def _(
    mesh: pv.PolyData, selection: Bool[ArrayLike, " C"] | Integer[ArrayLike, " N"]
) -> pv.PolyData:
    unstructured: pv.UnstructuredGrid = mesh.extract_cells(selection)  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = unstructured.extract_surface()  # pyright: ignore[reportAssignmentType]
    return surface


@extract_cells.register
def _(
    mesh: pv.UnstructuredGrid,
    selection: Bool[ArrayLike, " C"] | Integer[ArrayLike, " N"],
) -> pv.UnstructuredGrid:
    unstructured: pv.UnstructuredGrid = mesh.extract_cells(selection)  # pyright: ignore[reportAssignmentType]
    return unstructured
