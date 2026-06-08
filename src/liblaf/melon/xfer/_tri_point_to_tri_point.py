from collections.abc import Iterable
from typing import no_type_check

import einops
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Bool, Float, Integer

from liblaf.melon import io
from liblaf.melon.tri import fill_point


def tri_point_to_tri_point(
    source: pv.PolyData,
    target: pv.PolyData,
    names: Iterable[str] | None = None,
    *,
    max_dist: float = 0.01,
    normal_threshold: float = 0.8,
) -> pv.PolyData:
    if names is None:
        names: list[str] = source.point_data.keys()
    source_wp: wp.Mesh = io.as_warp_mesh(source)
    points: wp.array = wp.from_numpy(target.points, wp.vec3f)
    normals: wp.array = wp.from_numpy(target.point_normals, wp.vec3f)
    result_wp: wp.array = wp.zeros((target.n_points), wp.bool)
    face_wp: wp.array = wp.zeros((target.n_points), wp.int32)
    barycentric_wp: wp.array = wp.zeros((target.n_points,), wp.vec3f)
    wp.launch(
        _mesh_query_point_kernel,
        dim=(target.n_points,),
        inputs=[
            source_wp.id,
            points,
            normals,
            max_dist * source.length,
            normal_threshold,
        ],
        outputs=[result_wp, barycentric_wp, face_wp],
    )
    result: Bool[np.ndarray, " T"] = result_wp.numpy()
    face: Integer[np.ndarray, " T"] = face_wp.numpy()
    barycentric: Float[np.ndarray, "T 3"] = barycentric_wp.numpy()
    indices: Integer[np.ndarray, "T 3"] = source.regular_faces[face]
    for name in names:
        source_data: Float[np.ndarray, "S ..."] = source.point_data[name]
        target_data: Float[np.ndarray, "T ..."] = einops.einsum(
            barycentric, source_data[indices], "t i, t i ... -> t ..."
        )
        mask: Bool[np.ndarray, "T ..."] = result.reshape(
            (-1, *[1] * (target_data.ndim - 1))
        )
        target.point_data[name] = np.where(mask, target_data, np.nan)
    return target


@wp.kernel
@no_type_check
def _mesh_query_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    normals: wp.array1d[wp.vec3f],
    max_dist: wp.float32,
    normal_threshold: wp.float32,
    result: wp.array1d[wp.bool],
    barycentric: wp.array1d[wp.vec3f],
    face: wp.array1d[wp.int32],
) -> None:
    tid = wp.tid()  # int
    point = points[tid]  # vec3
    normal = normals[tid]  # vec3
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    if query.result:
        target_normal = wp.mesh_eval_face_normal(mesh_id, query.face)  # vec3
        normal_similarity = wp.dot(normal, target_normal)  # float
        if normal_similarity < normal_threshold:
            result[tid] = False
        else:
            result[tid] = True
            face[tid] = query.face
            barycentric[tid] = wp.vec3f(query.u, query.v, 1.0 - query.u - query.v)
    else:
        result[tid] = False
