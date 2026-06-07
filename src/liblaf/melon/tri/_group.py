from collections.abc import Iterable
from typing import Any

import more_itertools as mit
import numpy as np
import pyvista as pv
from jaxtyping import Bool
from pyvista import VectorLike

from liblaf.melon import io


def extract_cells(
    mesh: Any, ind: int | VectorLike[int] | VectorLike[bool], *, invert: bool = False
) -> pv.PolyData:
    """Extract selected cells from a surface-like mesh.

    Args:
        mesh: Object convertible to [`pyvista.PolyData`][pyvista.PolyData].
        ind: Cell index, integer indices, or boolean cell mask.
        invert: Extract all cells except the selected cells.

    Returns:
        Extracted surface.
    """
    mesh: pv.PolyData = io.as_polydata(mesh)
    ind: np.ndarray = np.asarray(ind)
    if np.isdtype(ind.dtype, "bool"):
        ind: np.ndarray = np.flatnonzero(ind)
    cells: pv.UnstructuredGrid = mesh.extract_cells(ind, invert=invert)
    surface: pv.PolyData = cells.extract_surface(algorithm=None)
    return surface


def extract_groups(
    mesh: Any, groups: int | str | Iterable[int | str], *, invert: bool = False
) -> pv.PolyData:
    """Extract cells whose `GroupId` matches numeric or named groups.

    Args:
        mesh: Mesh with `GroupId` cell data and optional `GroupName` field data.
        groups: Group id, group name, or iterable of ids and names.
        invert: Extract all cells outside the selected groups.

    Returns:
        Extracted surface.
    """
    return extract_cells(mesh, select_groups(mesh, groups), invert=invert)


def select_groups(
    mesh: Any,
    groups: int | str | Iterable[int | str],
    *,
    invert: bool = False,
    preference: pv.FieldAssociation = pv.FieldAssociation.CELL,
) -> Bool[np.ndarray, " cells"]:
    """Build a boolean mask for cells in selected groups.

    Args:
        mesh: Mesh with a group-id array.
        groups: Group id, group name, or iterable of ids and names.
        invert: Invert the selection mask.
        preference: PyVista association used to resolve the `GroupId` array.

    Returns:
        Boolean mask over mesh cells.
    """
    mesh: pv.PolyData = io.as_polydata(mesh)
    group_ids: list[int] = _as_group_ids(mesh, groups)
    mask: Bool[np.ndarray, " cells"] = np.isin(
        mesh.get_array("GroupId", preference),  # ty:ignore[invalid-argument-type]
        group_ids,
        invert=invert,
    )
    return mask


def _as_group_ids(
    mesh: pv.PolyData, groups: int | str | Iterable[int | str]
) -> list[int]:
    group_ids: list[int] = []
    for group in mit.always_iterable(groups, base_type=(int, str)):
        if isinstance(group, int):
            group_ids.append(group)
        elif isinstance(group, str):
            group_names: list[str] = mesh.field_data["GroupName"].tolist()
            group_ids.append(group_names.index(group))
        else:
            raise NotImplementedError
    return group_ids
