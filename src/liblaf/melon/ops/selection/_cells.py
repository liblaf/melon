from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import grapes


class GroupNamesKeyError(KeyError):
    def __init__(self) -> None:
        super().__init__("GroupNames")


def select_cells_by_group(
    mesh: Any, selection: int | str | Iterable[int | str] | None = None
) -> Bool[np.ndarray, " C"]:
    mesh: pv.PolyData = melon.as_poly_data(mesh)
    names: Sequence[str] | None = (
        mesh.field_data["GroupNames"].tolist()
        if "GroupNames" in mesh.field_data
        else None
    )
    selection: list[int] = _get_group_ids(selection, names)
    group_ids: Integer[np.ndarray, " C"] = mesh.cell_data["GroupIds"]
    selected: Bool[np.ndarray, " C"] = np.isin(group_ids, selection)
    return selected


def _get_group_ids(
    selection: int | str | Iterable[int | str] | None, names: Sequence[str] | None
) -> list[int]:
    if selection is None:
        if names is None:
            raise GroupNamesKeyError
        return list(range(len(names)))
    selection = grapes.as_iterable(selection)
    return [_get_group_id(group, names) for group in selection]


def _get_group_id(selection: int | str, names: Sequence[str] | None = None) -> int:
    if isinstance(selection, int):
        return selection
    if names is None:
        raise GroupNamesKeyError
    return names.index(selection)
