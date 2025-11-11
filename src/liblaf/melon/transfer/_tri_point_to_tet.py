import collections
from collections.abc import Iterable, Mapping
from typing import Any

import more_itertools as mit
import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf.melon import io
from liblaf.melon.proximity import NearestPointOnSurface


def transfer_tri_point_to_tet(
    source: Any,
    target: Any,
    *,
    data: str | Iterable[str] | None = None,
    fill: Any | Mapping[str, Any] | None = None,
    nearest: NearestPointOnSurface | None = None,  # noqa: ARG001
    point_id: str | None = None,
) -> pv.UnstructuredGrid:
    source: pv.PolyData = io.as_polydata(source)
    target: pv.UnstructuredGrid = io.as_unstructured_grid(target)
    data: Iterable[str] = _make_data_names(data, source)
    fill: Mapping[str, Any] = _make_fill_mapping(fill)
    result: pv.UnstructuredGrid
    if point_id is None:
        raise NotImplementedError
    result = _transfer_with_point_id(
        source, target, data=data, point_id=point_id, fill=fill
    )
    return result


def _make_data_names(
    data: str | Iterable[str] | None, source: pv.PolyData
) -> Iterable[str]:
    if data is None:
        return source.point_data.keys()
    return mit.always_iterable(data)


def _make_fill_mapping(
    fill: Any | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if isinstance(fill, Mapping):
        return fill
    return collections.defaultdict(lambda: fill)


def _transfer_with_point_id(
    source: pv.PolyData,
    target: pv.UnstructuredGrid,
    *,
    data: Iterable[str],
    fill: Mapping[str, Any],
    point_id: str,
) -> pv.UnstructuredGrid:
    target = target.copy()
    source_point_id: Integer[np.ndarray, " S"] = source.point_data[point_id]
    target_point_id: Integer[np.ndarray, " T"] = target.point_data[point_id]
    n_points: int = max(np.max(source_point_id), np.max(target_point_id)) + 1
    canonical_to_source: Integer[np.ndarray, " C"] = np.full(n_points, -1)
    canonical_to_source[source_point_id] = np.arange(source.n_points)
    canonical_to_target: Integer[np.ndarray, " C"] = np.full(n_points, -1)
    canonical_to_target[target_point_id] = np.arange(target.n_points)
    indices: Integer[np.ndarray, " S"] = canonical_to_target[source_point_id]
    for name in data:
        source_data: np.ndarray = source.point_data[name]
        target_data: np.ndarray = np.full(
            (target.n_points, *source_data.shape[1:]), fill[name], source_data.dtype
        )
        target_data[indices] = source_data
        target.point_data[name] = target_data
    return target
