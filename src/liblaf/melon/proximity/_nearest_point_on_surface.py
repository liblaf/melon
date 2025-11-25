from typing import Any, no_type_check, override

import attrs
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Float, Integer

from liblaf.melon import io

from ._abc import NearestAlgorithm, NearestAlgorithmPrepared, NearestResult


@attrs.define
class NearestPointOnSurfaceResult(NearestResult):
    barycentric: Float[np.ndarray, "N 3"]
    triangle_id: Integer[np.ndarray, " N"]


@attrs.frozen(kw_only=True)
class NearestPointOnSurfacePrepared(NearestAlgorithmPrepared):
    source_pv: pv.PolyData
    source: wp.Mesh

    distance_threshold: float
    ignore_orientation: bool
    normal_threshold: float | None

    @property
    def face_normals(self) -> Float[np.ndarray, "M 3"]:
        return self.source_pv.face_normals

    @property
    def length(self) -> float:
        return self.source_pv.length

    @override
    def query(self, query: Any) -> NearestPointOnSurfaceResult:
        if self.normal_threshold is None:
            return self._query_without_normal_threshold(query)
        return self._query_with_normal_threshold(query)

    def _query_without_normal_threshold(
        self, query: Any
    ) -> NearestPointOnSurfaceResult:
        query: pv.PointSet = io.as_pointset(query)
        n_points: int = query.n_points
        points_wp: wp.array = wp.from_numpy(query.points, wp.vec3f)
        barycentric: wp.array = wp.full((n_points,), wp.nan, wp.vec3f)  # pyright: ignore[reportArgumentType]
        distance: wp.array = wp.full((n_points,), wp.inf, wp.float32)  # pyright: ignore[reportArgumentType]
        missing: wp.array = wp.ones((n_points,), wp.bool)
        nearest: wp.array = wp.full((n_points,), wp.nan, wp.vec3f)  # pyright: ignore[reportArgumentType]
        triangle_id: wp.array = wp.full((n_points,), -1, wp.int32)  # pyright: ignore[reportArgumentType]
        wp.launch(
            _nearest_point_on_surface_kernel,
            dim=(n_points,),
            inputs=[self.source.id, points_wp, self.distance_threshold * self.length],
            outputs=[barycentric, distance, missing, nearest, triangle_id],
        )
        return NearestPointOnSurfaceResult(
            barycentric=barycentric.numpy(),
            distance=distance.numpy(),
            missing=missing.numpy(),
            nearest=nearest.numpy(),
            triangle_id=triangle_id.numpy(),
        )

    def _query_with_normal_threshold(self, query: Any) -> NearestPointOnSurfaceResult:
        query: pv.PointSet = io.as_pointset(query, point_normals=True)
        n_points: int = query.n_points
        points_wp: wp.array = wp.from_numpy(query.points, wp.vec3f)
        point_normals_wp: wp.array = wp.from_numpy(
            query.point_data["Normals"], wp.vec3f
        )
        barycentric: wp.array = wp.full((n_points,), wp.nan, wp.vec3f)  # pyright: ignore[reportArgumentType]
        distance: wp.array = wp.full((n_points,), wp.inf, wp.float32)  # pyright: ignore[reportArgumentType]
        missing: wp.array = wp.ones((n_points,), wp.bool)
        nearest: wp.array = wp.full((n_points,), wp.nan, wp.vec3f)  # pyright: ignore[reportArgumentType]
        triangle_id: wp.array = wp.full((n_points,), -1, wp.int32)  # pyright: ignore[reportArgumentType]
        wp.launch(
            _nearest_point_on_surface_with_normal_threshold_kernel,
            dim=(n_points,),
            inputs=[
                self.source.id,
                points_wp,
                point_normals_wp,
                self.distance_threshold * self.length,
                self.ignore_orientation,
                self.normal_threshold,
            ],
            outputs=[barycentric, distance, missing, nearest, triangle_id],
        )
        return NearestPointOnSurfaceResult(
            barycentric=barycentric.numpy(),
            distance=distance.numpy(),
            missing=missing.numpy(),
            nearest=nearest.numpy(),
            triangle_id=triangle_id.numpy(),
        )


@attrs.define(kw_only=True, on_setattr=attrs.setters.validate)
class NearestPointOnSurface(NearestAlgorithm):
    distance_threshold: float = 0.1
    ignore_orientation: bool = False
    normal_threshold: float | None = attrs.field(
        default=0.8,
        validator=attrs.validators.optional(
            [attrs.validators.ge(-1.0), attrs.validators.le(1.0)]
        ),
    )

    @override
    def prepare(self, source: Any) -> NearestPointOnSurfacePrepared:
        return NearestPointOnSurfacePrepared(
            source=io.as_warp_mesh(source),
            source_pv=io.as_polydata(source),
            distance_threshold=self.distance_threshold,
            ignore_orientation=self.ignore_orientation,
            normal_threshold=self.normal_threshold,
        )


def nearest_point_on_surface(
    source: Any,
    target: Any,
    *,
    distance_threshold: float = 0.1,
    ignore_orientation: bool = True,
    normal_threshold: float | None = 0.8,
) -> NearestPointOnSurfaceResult:
    algorithm = NearestPointOnSurface(
        distance_threshold=distance_threshold,
        ignore_orientation=ignore_orientation,
        normal_threshold=normal_threshold,
    )
    prepared: NearestPointOnSurfacePrepared = algorithm.prepare(source)
    return prepared.query(target)


@wp.kernel
@no_type_check
def _nearest_point_on_surface_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    distance_threshold: wp.float32,
    # outputs
    barycentric: wp.array(dtype=wp.vec3f),
    distance: wp.array(dtype=wp.float32),
    missing: wp.array(dtype=wp.bool),
    nearest: wp.array(dtype=wp.vec3f),
    triangle_id: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, point, distance_threshold)
    if query.result:
        missing[tid] = False
        barycentric[tid] = wp.vector(
            query.u, query.v, type(query.u)(1.0) - query.u - query.v
        )
        nearest[tid] = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance[tid] = wp.length(nearest[tid] - point)
        triangle_id[tid] = query.face
    else:
        missing[tid] = True


@wp.kernel
@no_type_check
def _nearest_point_on_surface_with_normal_threshold_kernel(  # noqa: PLR0913
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    point_normals: wp.array(dtype=wp.vec3f),
    distance_threshold: wp.float32,
    ignore_orientation: wp.bool,
    normal_threshold: wp.float32,
    # outputs
    barycentric: wp.array(dtype=wp.vec3f),
    distance: wp.array(dtype=wp.float32),
    missing: wp.array(dtype=wp.bool),
    nearest: wp.array(dtype=wp.vec3f),
    triangle_id: wp.array(dtype=wp.int32),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, point, distance_threshold)
    if query.result:
        face_normal = wp.mesh_eval_face_normal(mesh_id, query.face)
        cosine_similarity = wp.dot(face_normal, point_normals[tid]) / (
            wp.length(face_normal) * wp.length(point_normals[tid])
        )
        if ignore_orientation:
            cosine_similarity = wp.abs(cosine_similarity)
        if cosine_similarity >= normal_threshold:
            missing[tid] = False
            barycentric[tid] = wp.vector(
                query.u, query.v, type(query.u)(1.0) - query.u - query.v
            )
            nearest[tid] = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            distance[tid] = wp.length(nearest[tid] - point)
            triangle_id[tid] = query.face
        else:
            missing[tid] = True
    else:
        missing[tid] = True
