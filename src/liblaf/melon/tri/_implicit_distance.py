from collections.abc import Sequence
from typing import no_type_check

import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.melon.utils import warp_stream_from_torch


def implicit_distance(
    mesh: wp.Mesh, points: Float[Tensor, "*q 3"], max_dist: float = wp.inf
) -> Float[Tensor, "*q"]:
    shape: Sequence[int] = points.shape[:-1]
    points: Float[Tensor, "q 3"] = points.reshape(-1, 3)
    n_points: int = points.shape[0]
    points_wp: wp.array = wp.from_torch(points, wp.vec3f, return_ctype=True)
    distance_wp: wp.array = wp.empty((n_points,), wp.float32)
    wp.launch(
        _mesh_query_point_kernel,
        (n_points,),
        inputs=[mesh.id, points_wp, max_dist],
        outputs=[distance_wp],
        stream=warp_stream_from_torch(),
    )
    distance: Float[Tensor, " q"] = wp.to_torch(distance_wp)
    distance: Float[Tensor, "*q"] = distance.reshape(shape)
    return distance


@wp.kernel
@no_type_check
def _mesh_query_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    max_dist: wp.float32,
    distance: wp.array1d[wp.float32],
) -> None:
    tid = wp.tid()
    point = points[tid]  # vec3
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    if query.result:
        position = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)  # vec3
        dist = wp.length(position - point)
        if query.sign < 0:
            dist = -dist
        distance[tid] = dist
    else:
        distance[tid] = wp.inf
