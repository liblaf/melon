from collections.abc import Sequence
from typing import no_type_check

import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.melon.utils import warp_stream_from_torch


def query_ray(
    mesh: wp.Mesh,
    start: Float[Tensor, "*q 3"],
    direction: Float[Tensor, "*q 3"],
    max_t: float = wp.inf,
) -> Float[Tensor, "*q"]:
    shape: Sequence[int] = start.shape[:-1]
    start: Float[Tensor, "q 3"] = start.reshape(-1, 3)
    direction: Float[Tensor, "q 3"] = direction.reshape(-1, 3)
    start_wp: wp.array = wp.from_torch(start, wp.vec3f, return_ctype=True)
    direction_wp: wp.array = wp.from_torch(direction, wp.vec3f, return_ctype=True)
    distance_wp: wp.array = wp.empty((start.shape[0],), wp.float32)
    wp.launch(
        _mesh_query_ray_kernel,
        dim=(start.shape[0],),
        inputs=[mesh.id, start_wp, direction_wp, max_t],
        outputs=[distance_wp],
        stream=warp_stream_from_torch(),
    )
    distance: Float[Tensor, " q"] = wp.to_torch(distance_wp)
    distance: Float[Tensor, "*q"] = distance.reshape(shape)
    return distance


@wp.kernel
@no_type_check
def _mesh_query_ray_kernel(
    mesh_id: wp.uint64,
    start: wp.array1d[wp.vec3f],
    direction: wp.array1d[wp.vec3f],
    max_t: wp.float32,
    distance: wp.array1d[wp.float32],
) -> None:
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh_id, start[tid], direction[tid], max_t)
    if query.result:
        distance[tid] = query.t
    else:
        distance[tid] = wp.inf
