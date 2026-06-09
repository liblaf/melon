import functools
from typing import no_type_check

import attrs
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Bool, Float


@attrs.frozen
class TetraSurfaceBroadPhaseResult:
    distance: Float[np.ndarray, " c"]
    radius: Float[np.ndarray, " c"]

    @functools.cached_property
    def boundary(self) -> Bool[np.ndarray, " c"]:
        return ~(self.inside | self.outside)

    @functools.cached_property
    def inside(self) -> Bool[np.ndarray, " c"]:
        return self.distance < -self.radius

    @functools.cached_property
    def outside(self) -> Bool[np.ndarray, " c"]:
        return self.distance > self.radius


def tetra_surface_broad_phase(
    centers: pv.PolyData, surface: wp.Mesh
) -> TetraSurfaceBroadPhaseResult:
    radius: Float[np.ndarray, " c"] = np.array(centers.point_data["Radius"])
    points_wp: wp.array = wp.from_numpy(centers.points, wp.vec3f)
    distance_wp: wp.array = wp.zeros((centers.n_points,), wp.float32)
    wp.launch(
        _mesh_query_point_kernel,
        dim=(centers.n_points,),
        inputs=[surface.id, points_wp, wp.inf],
        outputs=[distance_wp],
    )
    distance: Float[np.ndarray, " c"] = distance_wp.numpy()
    return TetraSurfaceBroadPhaseResult(distance=distance, radius=radius)


@wp.kernel
@no_type_check
def _mesh_query_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    max_dist: wp.float32,
    distance: wp.array1d[wp.float32],
) -> None:
    tid = wp.tid()  # int
    point = points[tid]  # vec3
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    if query.result:
        position = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)  # vec3
        dist = wp.length(position - point)  # float
        if query.sign < 0:
            dist = -dist
        distance[tid] = dist
    else:
        distance[tid] = max_dist
