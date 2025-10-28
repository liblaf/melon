import collections
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf import grapes
from liblaf.melon import io
from liblaf.melon.proximity import NearestPoint
from liblaf.melon.tri import transfer_point_data


def transfer_point_data_from_surface(
    src: Any,
    dst: Any,
    *,
    data: str | Iterable[str] | None = None,
    fill: Any | Mapping[str, Any] | None = None,
    nearest: NearestPoint | None = None,
) -> pv.UnstructuredGrid:
    src: pv.PolyData = io.as_polydata(src)
    dst: pv.UnstructuredGrid = io.as_unstructured_grid(dst).copy()
    dst.point_data["__point_id"] = np.arange(dst.n_points)
    data = src.point_data.keys() if data is None else grapes.as_iterable(data)
    fill = _make_fill(fill)
    surface: pv.PolyData = dst.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface = transfer_point_data(src, surface, data=data, fill=fill, nearest=nearest)
    surface_point_id: Integer[np.ndarray, " N"] = surface.point_data["__point_id"]
    for name in data:
        surface_data: np.ndarray = surface.point_data[name]
        if name not in dst.point_data:
            dst.point_data[name] = np.full(
                (dst.n_points, *surface_data.shape[1:]), fill[name]
            )
        dst.point_data[name][surface_point_id] = surface_data
    del dst.point_data["__point_id"]
    return dst


def _make_fill(fill: Any) -> Mapping[str, Any]:
    if isinstance(fill, Mapping):
        return fill
    return collections.defaultdict(lambda: fill)
