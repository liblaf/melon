from collections.abc import Iterable
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

from liblaf import grapes
from liblaf.melon import io


def select_groups(
    mesh: Any, groups: int | str | Iterable[int | str]
) -> Integer[np.ndarray, " N"]:
    mesh: pv.PolyData = io.as_poly_data(mesh)
    group_ids: list[int] = as_group_ids(mesh, groups)
    mask: Bool[np.ndarray, " C"] = np.isin(mesh.cell_data["GroupIds"], group_ids)
    indices: Integer[np.ndarray, " N"]
    (indices,) = np.nonzero(mask)
    return indices


def as_group_ids(
    mesh: pv.PolyData, groups: int | str | Iterable[int | str]
) -> list[int]:
    groups = grapes.as_iterable(groups)
    group_ids: list[int] = []
    for group in groups:
        if isinstance(group, int):
            group_ids.append(group)
        elif isinstance(group, str):
            group_names: list[str] = list(mesh.field_data["GroupNames"])
            group_ids.append(group_names.index(group))
        else:
            raise NotImplementedError
    return group_ids
