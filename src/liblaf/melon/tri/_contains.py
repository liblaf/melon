from collections.abc import Sequence
from typing import no_type_check

import warp as wp
from jaxtyping import Bool, Float
from torch import Tensor

from liblaf.melon.utils import warp_stream_from_torch


def contains(
    mesh: wp.Mesh, points: Float[Tensor, "*p 3"], max_dist: float = wp.inf
) -> Bool[Tensor, "*p"]:
    shape: Sequence[int] = points.shape[:-1]
    points: Float[Tensor, "p 3"] = points.reshape(-1, 3)
    n_points: int = points.shape[0]
    points_wp: wp.array = wp.from_torch(
        points.contiguous(), wp.vec3f, return_ctype=True
    )
    contains_wp: wp.array = wp.empty((n_points,), wp.bool)
    wp.launch(
        _mesh_query_point_kernel,
        dim=(n_points,),
        inputs=[mesh.id, points_wp, max_dist],
        outputs=[contains_wp],
        stream=warp_stream_from_torch(),
    )
    contains: Bool[Tensor, " p"] = wp.to_torch(contains_wp)
    contains: Bool[Tensor, "*p"] = contains.reshape(shape)
    return contains


@wp.kernel
@no_type_check
def _mesh_query_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    max_dist: wp.float32,
    contains: wp.array1d[wp.bool],
) -> None:
    tid = wp.tid()
    point = points[tid]  # vec3
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    if query.result:
        contains[tid] = query.sign < 0
    else:
        contains[tid] = False
