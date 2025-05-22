import functools
from typing import Any, override

import attrs
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._abc import NearestAlgorithm, NearestAlgorithmPrepared, NearestResult
from ._nearest_point import NearestPoint, NearestPointPrepared, NearestPointResult


class NearestPointOnSurfaceResult(NearestResult):
    triangle_id: Integer[np.ndarray, " N"]


@attrs.frozen(kw_only=True)
class NearestPointOnSurfacePrepared(NearestAlgorithmPrepared):
    source: tm.Trimesh

    distance_threshold: float
    fallback_to_nearest_vertex: bool
    ignore_orientation: bool
    max_k: int
    normal_threshold: float
    workers: int

    @override
    def query(self, query: Any) -> NearestPointOnSurfaceResult:
        need_normals: bool = self.normal_threshold > -1.0
        query: pv.PointSet = io.as_point_set(query, point_normals=need_normals)
        nearest: Float[np.ndarray, "N 3"]
        distance: Float[np.ndarray, " N"]
        triangle_id: Integer[np.ndarray, " N"]
        nearest, distance, triangle_id = self.source.nearest.on_surface(query.points)
        missing: Bool[np.ndarray, " N"] = (
            distance > self.distance_threshold * self.source.scale
        )
        if need_normals:
            source_normals: Float[np.ndarray, "N 3"] = self.source.face_normals[
                triangle_id
            ]
            target_normals: Float[np.ndarray, "N 3"] = query.point_data["Normals"]
            cosine_similarity: Float[np.ndarray, " N"] = np.vecdot(
                source_normals, target_normals
            )
            if self.ignore_orientation:
                cosine_similarity = np.abs(cosine_similarity)
            missing |= cosine_similarity < self.normal_threshold
        distance[missing] = np.inf
        nearest[missing] = np.nan
        triangle_id[missing] = -1
        result = NearestPointOnSurfaceResult(
            distance=distance, missing=missing, nearest=nearest, triangle_id=triangle_id
        )
        if self.fallback_to_nearest_vertex:
            result = self._fallback_to_nearest_vertex(query, result)
        return result

    @functools.cached_property
    def _nearest_vertex(self) -> NearestPointPrepared:
        return NearestPoint(
            distance_threshold=self.distance_threshold,
            max_k=self.max_k,
            normal_threshold=self.normal_threshold,
            workers=self.workers,
        ).prepare(self.source)

    def _fallback_to_nearest_vertex(
        self, query: pv.PointSet, result: NearestPointOnSurfaceResult
    ) -> NearestPointOnSurfaceResult:
        missing_vid: Integer[np.ndarray, " N"] = result["missing"].nonzero()[0]
        remaining: pv.PointSet = query.extract_points(missing_vid, include_cells=False)  # pyright: ignore[reportAssignmentType]
        remaining_result: NearestPointResult = self._nearest_vertex.query(remaining)
        result["distance"][missing_vid] = remaining_result["distance"]
        result["missing"][missing_vid] = remaining_result["missing"]
        result["nearest"][missing_vid] = remaining_result["nearest"]
        result["triangle_id"][missing_vid] = self._vertex_id_to_triangle_id(
            remaining_result["vertex_id"]
        )
        return result

    def _vertex_id_to_triangle_id(
        self, vertex_id: Integer[ArrayLike, " N"]
    ) -> Integer[np.ndarray, " N"]:
        return self.source.vertex_faces[vertex_id, 0]


@attrs.define(kw_only=True, on_setattr=attrs.setters.validate)
class NearestPointOnSurface(NearestAlgorithm):
    distance_threshold: float = 0.1
    fallback_to_nearest_vertex: bool = True
    ignore_orientation: bool = True
    max_k: int = 32
    normal_threshold: float = attrs.field(
        default=0.8, validator=attrs.validators.le(1.0)
    )
    workers: int = -1

    @override
    def prepare(self, source: Any) -> NearestPointOnSurfacePrepared:
        source: tm.Trimesh = io.as_trimesh(source)
        return NearestPointOnSurfacePrepared(
            distance_threshold=self.distance_threshold,
            fallback_to_nearest_vertex=self.fallback_to_nearest_vertex,
            ignore_orientation=self.ignore_orientation,
            max_k=self.max_k,
            normal_threshold=self.normal_threshold,
            workers=self.workers,
            source=source,
        )


def nearest_point_on_surface(
    source: Any,
    target: Any,
    *,
    distance_threshold: float = 0.1,
    fallback_to_nearest_vertex: bool = True,
    ignore_orientation: bool = True,
    max_k: int = 32,
    normal_threshold: float = 0.8,
    workers: int = -1,
) -> NearestPointOnSurfaceResult:
    algorithm = NearestPointOnSurface(
        distance_threshold=distance_threshold,
        fallback_to_nearest_vertex=fallback_to_nearest_vertex,
        ignore_orientation=ignore_orientation,
        max_k=max_k,
        normal_threshold=normal_threshold,
        workers=workers,
    )
    prepared: NearestPointOnSurfacePrepared = algorithm.prepare(source)
    return prepared.query(target)
