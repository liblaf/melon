from typing import Any, override

import attrs
import numpy as np
import pyvista as pv
import scipy.spatial
from jaxtyping import Bool, Float, Integer

from liblaf.melon import io

from ._abc import NearestAlgorithm, NearestAlgorithmPrepared, NearestResult


@attrs.define
class NearestPointResult(NearestResult):
    vertex_id: Integer[np.ndarray, " N"]


@attrs.frozen(kw_only=True)
class NearestPointPrepared(NearestAlgorithmPrepared):
    source: pv.PointSet
    tree: scipy.spatial.KDTree

    distance_threshold: float
    ignore_orientation: bool
    max_k: int
    normal_threshold: float
    workers: int

    @override
    def query(self, query: Any) -> NearestPointResult:
        if self.normal_threshold <= -1.0:
            return self._nearest_vertex(query)
        return self._nearest_vertex_with_normal_threshold(query)

    def _nearest_vertex(self, query: Any) -> NearestPointResult:
        query: pv.PointSet = io.as_pointset(query)
        distance: Float[np.ndarray, " N"]
        vertex_id: Integer[np.ndarray, " N"]
        distance, vertex_id = self.tree.query(
            query.points,
            distance_upper_bound=self.distance_threshold * self.source.length,
            workers=self.workers,
        )  # pyright: ignore[reportAssignmentType]
        missing: Bool[np.ndarray, " N"] = vertex_id == self.source.n_points
        distance[missing] = np.inf
        vertex_id[missing] = -1
        nearest: Float[np.ndarray, " N 3"] = self.source.points[vertex_id]
        nearest[missing] = np.nan
        return NearestPointResult(
            distance=distance, missing=missing, nearest=nearest, vertex_id=vertex_id
        )

    def _nearest_vertex_with_normal_threshold(self, query: Any) -> NearestPointResult:
        source_normals: Float[np.ndarray, "N 3"] = self.source.point_data["Normals"]
        query: pv.PointSet = io.as_pointset(query, point_normals=True)
        query_normals: Float[np.ndarray, "N 3"] = query.point_data["Normals"]
        result: NearestPointResult = self._nearest_vertex(query)
        distance: Float[np.ndarray, " N"] = result.distance
        missing: Bool[np.ndarray, " N"] = result.missing
        nearest: Float[np.ndarray, " N 3"] = result.nearest
        vertex_id: Integer[np.ndarray, " N"] = result.vertex_id
        remaining_vertex_id: Integer[np.ndarray, " R"] = missing.nonzero()[0]
        k: int = 2
        while k <= self.max_k and remaining_vertex_id.size > 0:
            d: Float[np.ndarray, "R k"]
            v: Integer[np.ndarray, "R k"]
            d, v = self.tree.query(
                query.points[remaining_vertex_id],
                k=k,
                distance_upper_bound=self.distance_threshold * self.source.length,
                workers=self.workers,
            )  # pyright: ignore[reportAssignmentType]
            next_remaining_vertex_id: list[int] = []
            for i, vid in enumerate(remaining_vertex_id):
                for j in range(k):
                    if v[i, j] == self.source.n_points:
                        continue
                    cosine_similarity: float = np.dot(
                        source_normals[v[i, j]], query_normals[vid]
                    )
                    if self.ignore_orientation:
                        cosine_similarity = np.abs(cosine_similarity)
                    if cosine_similarity < self.normal_threshold:
                        continue
                    distance[vid] = d[i, j]
                    missing[vid] = False
                    vertex_id[vid] = v[i, j]
                    nearest[vid] = self.source.points[v[i, j]]
                    break
                else:
                    next_remaining_vertex_id.append(vid)
            remaining_vertex_id = np.asarray(next_remaining_vertex_id)
            k *= 2
        return NearestPointResult(
            distance=distance, missing=missing, nearest=nearest, vertex_id=vertex_id
        )


@attrs.define(kw_only=True, on_setattr=attrs.setters.validate)
class NearestPoint(NearestAlgorithm):
    distance_threshold: float = 0.1
    ignore_orientation: bool = True
    max_k: int = 32
    normal_threshold: float = attrs.field(
        default=-np.inf, validator=attrs.validators.le(1.0)
    )
    workers: int = -1

    @override
    def prepare(self, source: Any) -> NearestPointPrepared:
        need_normals: bool = self.normal_threshold > -1.0
        source: pv.PointSet = io.as_pointset(source, point_normals=need_normals)
        tree: scipy.spatial.KDTree = scipy.spatial.KDTree(source.points)
        return NearestPointPrepared(
            source=source,
            tree=tree,
            distance_threshold=self.distance_threshold,
            max_k=self.max_k,
            normal_threshold=self.normal_threshold,
            ignore_orientation=self.ignore_orientation,
            workers=self.workers,
        )
