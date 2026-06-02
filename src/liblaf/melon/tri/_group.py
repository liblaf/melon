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
    return extract_cells(mesh, select_groups(mesh, groups), invert=invert)


def select_groups(
    mesh: Any, groups: int | str | Iterable[int | str], *, invert: bool = False
) -> Bool[np.ndarray, " cells"]:
    mesh: pv.PolyData = io.as_polydata(mesh)
    group_ids: list[int] = _as_group_ids(mesh, groups)
    mask: Bool[np.ndarray, " cells"] = np.isin(
        mesh.cell_data["GroupId"], group_ids, invert=invert
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
