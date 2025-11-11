from typing import Any, override

import attrs
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float, Integer

from liblaf.melon import io

from ._abc import NearestAlgorithm, NearestAlgorithmPrepared, NearestResult


@attrs.define
class NearestPointOnSurfaceResult(NearestResult):
    triangle_id: Integer[np.ndarray, " N"]


@attrs.frozen(kw_only=True)
class NearestPointOnSurfacePrepared(NearestAlgorithmPrepared):
    source: tm.Trimesh

    distance_threshold: float
    fallback: bool
    ignore_orientation: bool
    normal_threshold: float

    @override
    def query(self, query: Any) -> NearestPointOnSurfaceResult:
        need_normals: bool = self.normal_threshold > -1.0
        query: pv.PointSet = io.as_pointset(query, point_normals=need_normals)
        nearest: Float[np.ndarray, "N 3"]
        distance: Float[np.ndarray, " N"]
        triangle_id: Integer[np.ndarray, " N"]
        nearest, distance, triangle_id = self.source.nearest.on_surface(query.points)
        missing_distance: Bool[np.ndarray, " N"] = (
            distance > self.distance_threshold * self.source.scale
        )
        distance[missing_distance] = np.inf
        nearest[missing_distance] = np.nan
        triangle_id[missing_distance] = -1
        result: NearestPointOnSurfaceResult = NearestPointOnSurfaceResult(
            distance=distance,
            missing=missing_distance,
            nearest=nearest,
            triangle_id=triangle_id,
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
            missing_normal: Bool[np.ndarray, " N"] = (
                cosine_similarity < self.normal_threshold
            )
            result.distance[missing_normal] = np.inf
            result.missing |= missing_normal
            result.nearest[missing_normal] = np.nan
            result.triangle_id[missing_normal] = -1
            if self.fallback:
                result = self._fallback(
                    query, result, ~missing_distance & missing_normal
                )
        return result

    def _fallback(
        self,
        query: pv.PointSet,
        result: NearestPointOnSurfaceResult,
        missing: Bool[np.ndarray, " M"],
    ) -> NearestPointOnSurfaceResult:
        indices: Integer[np.ndarray, " M"] = np.flatnonzero(missing)
        for idx in indices:
            point: Float[np.ndarray, " 3"] = query.points[idx]
            point_normal: Float[np.ndarray, " 3"] = query.point_data["Normals"][idx]
            cosine_similarity: Float[np.ndarray, " S"] = np.vecdot(
                self.source.face_normals, point_normal[np.newaxis]
            )
            if self.ignore_orientation:
                cosine_similarity = np.abs(cosine_similarity)
            face_mask: Bool[np.ndarray, " S"] = (
                cosine_similarity >= self.normal_threshold
            )
            submesh: tm.Trimesh
            (submesh,) = self.source.submesh([face_mask])  # pyright: ignore[reportGeneralTypeIssues]
            nearest: Float[np.ndarray, " 1 3"]
            distance: Float[np.ndarray, " 1"]
            triangle_id: Integer[np.ndarray, " 1"]
            nearest, distance, triangle_id = submesh.nearest.on_surface(
                point[np.newaxis, :]
            )
            if distance[0] <= self.distance_threshold * self.source.scale:
                result.distance[idx] = distance[0]
                result.missing[idx] = False
                result.nearest[idx] = nearest[0]
                result.triangle_id[idx] = triangle_id[0]
        return result


@attrs.define(kw_only=True, on_setattr=attrs.setters.validate)
class NearestPointOnSurface(NearestAlgorithm):
    distance_threshold: float = 0.1
    fallback: bool = True
    ignore_orientation: bool = False
    normal_threshold: float = attrs.field(
        default=0.8, validator=attrs.validators.le(1.0)
    )

    @override
    def prepare(self, source: Any) -> NearestPointOnSurfacePrepared:
        source: tm.Trimesh = io.as_trimesh(source)
        return NearestPointOnSurfacePrepared(
            distance_threshold=self.distance_threshold,
            fallback=self.fallback,
            ignore_orientation=self.ignore_orientation,
            normal_threshold=self.normal_threshold,
            source=source,
        )


def nearest_point_on_surface(
    source: Any,
    target: Any,
    *,
    distance_threshold: float = 0.1,
    fallback: bool = True,
    ignore_orientation: bool = True,
    normal_threshold: float = 0.8,
) -> NearestPointOnSurfaceResult:
    algorithm = NearestPointOnSurface(
        distance_threshold=distance_threshold,
        fallback=fallback,
        ignore_orientation=ignore_orientation,
        normal_threshold=normal_threshold,
    )
    prepared: NearestPointOnSurfacePrepared = algorithm.prepare(source)
    return prepared.query(target)
