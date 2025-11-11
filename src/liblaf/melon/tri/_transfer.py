import collections
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

from liblaf import grapes
from liblaf.melon import io
from liblaf.melon.proximity import (
    NearestPoint,
    NearestPointOnSurface,
    NearestPointOnSurfacePrepared,
    NearestPointOnSurfaceResult,
    NearestPointPrepared,
    NearestPointResult,
)


def transfer_point_data(
    src: Any,
    dst: Any,
    *,
    data: str | Iterable[str] | None = None,
    fill: Any | Mapping[str, Any] | None = None,
    nearest: NearestPoint | None = None,
) -> pv.PolyData:
    src: pv.PolyData = io.as_polydata(src)
    dst: pv.PolyData = io.as_polydata(dst)
    data = src.point_data.keys() if data is None else grapes.as_iterable(data)
    fill = fill if isinstance(fill, Mapping) else collections.defaultdict(lambda: fill)
    nearest = NearestPoint() if nearest is None else nearest
    prepared: NearestPointPrepared = nearest.prepare(src)
    query: NearestPointResult = prepared.query(dst)
    missing: Bool[np.ndarray, " N"] = query.missing
    point_id: Integer[np.ndarray, " N"] = query.vertex_id
    for name in data:
        dst.point_data[name] = src.point_data[name][point_id]
        if np.any(missing):
            dst.point_data[name][missing] = fill[name]
    return dst


def transfer_cell_data_to_point(
    src: Any,
    dst: Any,
    *,
    data: str | Iterable[str] | None = None,
    fill: Any | Mapping[str, Any] | None = None,
    nearest: NearestPointOnSurface | None = None,
) -> pv.PolyData:
    src: pv.PolyData = io.as_polydata(src)
    src = src.triangulate()  # pyright: ignore[reportAssignmentType]
    dst: pv.PolyData = io.as_polydata(dst)
    data = src.cell_data.keys() if data is None else grapes.as_iterable(data)
    fill = fill if isinstance(fill, Mapping) else collections.defaultdict(lambda: fill)
    nearest = NearestPointOnSurface() if nearest is None else nearest
    prepared: NearestPointOnSurfacePrepared = nearest.prepare(src)
    query: NearestPointOnSurfaceResult = prepared.query(dst.points)
    missing: Bool[np.ndarray, " N"] = query.missing
    triangle_id: Integer[np.ndarray, " N"] = query.triangle_id
    for name, arr in src.cell_data.items():
        dst.point_data[name] = arr[triangle_id]
        if np.any(missing):
            dst.point_data[name][missing] = fill[name]
    return dst
