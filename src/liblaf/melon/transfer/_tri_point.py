import collections
from collections.abc import Iterable, Mapping
from typing import Any

import einops
import more_itertools as mit
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float, Integer

from liblaf.melon import io
from liblaf.melon.proximity import (
    NearestPointOnSurface,
    NearestPointOnSurfacePrepared,
    NearestPointOnSurfaceResult,
)


def transfer_tri_point(
    source: Any,
    target: Any,
    *,
    data: str | Iterable[str] | None = None,
    fill: Any | Mapping[str, Any] | None = None,
    nearest: NearestPointOnSurface | None = None,
) -> Any:
    source: pv.PolyData = io.as_polydata(source)
    target: pv.PolyData = io.as_polydata(target)
    data: Iterable[str] = _make_data_names(data, source)
    fill = _make_fill_mapping(fill)
    if nearest is None:
        nearest = NearestPointOnSurface()
    prepared: NearestPointOnSurfacePrepared = nearest.prepare(source)
    result: NearestPointOnSurfaceResult = prepared.query(target)
    any_missing: bool = np.any(result.missing)  # pyright: ignore[reportAssignmentType]
    valid: Bool[np.ndarray, " T"] = ~result.missing
    indices: Integer[np.ndarray, "V 3"] = source.regular_faces[
        result.triangle_id[valid]
    ]
    barycentric: Float[np.ndarray, "V 3"] = tm.triangles.points_to_barycentric(
        source.points[indices], result.nearest[valid]
    )
    for name in data:
        source_data: Float[np.ndarray, "S ..."] = source.point_data[name]
        target_data_valid: Float[np.ndarray, "V ..."] = einops.einsum(
            barycentric, source_data[indices], "V B, V B ... -> V ..."
        )
        target_data: Float[np.ndarray, "T ..."]
        if any_missing:
            target_data = np.full(
                (target.n_points, *source_data.shape[1:]), fill[name], source_data.dtype
            )
            target_data[valid] = target_data_valid
        else:
            target_data = target_data_valid
        target.point_data[name] = target_data
    return target


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
