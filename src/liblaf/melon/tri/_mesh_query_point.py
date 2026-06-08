from collections.abc import Sequence
from typing import no_type_check

import attrs
import torch
import warp as wp
from jaxtyping import Bool, Float, Integer
from torch import Tensor


@attrs.frozen
class MeshQueryPointResult:
    result: Bool[Tensor, "*Q"]
    face: Integer[Tensor, "*Q"]
    distance: Float[Tensor, "*Q"]


def mesh_query_point(
    mesh: wp.Mesh, points: Float[Tensor, "*Q 3"], max_dist: float
) -> MeshQueryPointResult:
    shape: Sequence[int] = points.shape[:-1]
    points: Float[Tensor, "Q 3"] = points.reshape(-1, 3)
    n_points: int = points.shape[0]
    points_wp: wp.array = wp.from_torch(points, wp.vec3f, return_ctype=True)
    result_wp: wp.array = wp.zeros((n_points,), wp.bool)
    face_wp: wp.array = wp.zeros((n_points,), wp.int32)
    distance_wp: wp.array = wp.zeros((n_points,), wp.float32)
    with _stream():
        wp.launch(
            _mesh_query_point_kernel,
            dim=(n_points,),
            inputs=[mesh.id, points_wp, max_dist],
            outputs=[result_wp, face_wp, distance_wp],
        )
    result: Bool[Tensor, "*Q"] = torch.reshape(wp.to_torch(result_wp), shape)
    face: Integer[Tensor, "*Q"] = torch.reshape(wp.to_torch(face_wp), shape)
    distance: Float[Tensor, "*Q"] = torch.reshape(wp.to_torch(distance_wp), shape)
    return MeshQueryPointResult(result=result, face=face, distance=distance)


@wp.kernel
@no_type_check
def _mesh_query_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    max_dist: wp.float32,
    result: wp.array1d[wp.bool],
    face: wp.array1d[wp.int32],
    distance: wp.array1d[wp.float32],
) -> None:
    tid = wp.tid()
    point = points[tid]  # vec3
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    result[tid] = query.result
    if query.result:
        face[tid] = query.face
        position = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)  # vec3
        dist = wp.length(position - point)
        if query.sign < 0:
            dist = -dist
        distance[tid] = dist


def _stream() -> wp.ScopedStream:
    if not torch.cuda.is_available():
        return wp.ScopedStream(None)
    stream_torch: torch.cuda.Stream = torch.cuda.current_stream()
    stream_wp: wp.Stream = wp.stream_from_torch(stream_torch)
    return wp.ScopedStream(stream_wp)
