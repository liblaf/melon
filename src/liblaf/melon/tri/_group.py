import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

from liblaf import grapes
from liblaf.melon import io

logger: logging.Logger = logging.getLogger(__name__)


def select_groups(
    mesh: Any, groups: int | str | Iterable[int | str], *, invert: bool = False
) -> Bool[np.ndarray, " cells"]:
    mesh: pv.PolyData = io.as_polydata(mesh)
    group_ids: list[int] = as_group_ids(mesh, groups)
    mask: Bool[np.ndarray, " C"] = np.isin(
        _get_group_id(mesh), group_ids, invert=invert
    )
    return mask


def as_group_ids(
    mesh: pv.PolyData, groups: int | str | Iterable[int | str]
) -> list[int]:
    groups = grapes.as_iterable(groups)
    group_ids: list[int] = []
    for group in groups:
        if isinstance(group, int):
            group_ids.append(group)
        elif isinstance(group, str):
            group_names: list[str] = _get_group_name(mesh).tolist()
            group_ids.append(group_names.index(group))
        else:
            raise NotImplementedError
    return group_ids


def _get_group_id(mesh: pv.PolyData) -> Integer[np.ndarray, " cell"]:
    key: str = "GroupIds"
    if key in mesh.cell_data:
        return mesh.cell_data[key]
    for key in ["group_id", "group_ids", "group-id", "group-ids", "GroupId"]:
        if key in mesh.cell_data:
            logger.warning(
                "'%s' is deprecated. Use 'GroupIds' instead.", key, extra={"once": True}
            )
            return mesh.cell_data[key]
    key = "GroupIds"
    raise KeyError(key)


def _get_group_name(mesh: pv.PolyData) -> np.ndarray:
    key: str = "GroupNames"
    if key in mesh.field_data:
        return mesh.field_data[key]
    for key in ["group_name", "group_names", "group-name", "group-names", "GroupName"]:
        if key in mesh.field_data:
            logger.warning(
                "'%s' is deprecated. Use 'GroupNames' instead.",
                key,
                extra={"once": True},
            )
            return mesh.field_data[key]
    key = "GroupNames"
    raise KeyError(key)
