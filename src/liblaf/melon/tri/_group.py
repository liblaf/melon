from collections.abc import Iterable
from typing import Any

import more_itertools as mit
import numpy as np
import pyvista as pv
from jaxtyping import Bool

from liblaf.melon import compat, io


def select_groups(
    mesh: Any, groups: int | str | Iterable[int | str], *, invert: bool = False
) -> Bool[np.ndarray, " cells"]:
    mesh: pv.PolyData = io.as_polydata(mesh)
    group_ids: list[int] = as_group_ids(mesh, groups)
    mask: Bool[np.ndarray, " cells"] = np.isin(
        compat.get_group_id(mesh), group_ids, invert=invert
    )
    return mask


def as_group_ids(
    mesh: pv.PolyData, groups: int | str | Iterable[int | str]
) -> list[int]:
    group_ids: list[int] = []
    for group in mit.always_iterable(groups, base_type=(int, str)):
        if isinstance(group, int):
            group_ids.append(group)
        elif isinstance(group, str):
            group_names: list[str] = compat.get_group_name(mesh)
            group_ids.append(group_names.index(group))
        else:
            raise NotImplementedError
    return group_ids
