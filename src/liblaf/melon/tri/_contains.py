import functools
from typing import Any, no_type_check

import attrs
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Array, Bool, Float

from liblaf.melon import io
from liblaf.melon._src.bounds import bounds_contains


@attrs.define
class MeshContainsPoints:
    mesh: Any

    @property
    def bounds(self) -> Float[np.ndarray, "2 3"]:
        return self.mesh_tm.bounds

    @functools.cached_property
    def mesh_tm(self) -> tm.Trimesh:
        return io.as_trimesh(self.mesh)

    @functools.cached_property
    def mesh_wp(self) -> wp.Mesh:
        return io.as_warp_mesh(self.mesh)

    @property
    def scale(self) -> float:
        return self.mesh_tm.scale

    def contains(self, pcl: Any) -> Bool[Array, " N"]:
        pcl: pv.PointSet = io.as_pointset(pcl)
        points_jax: Float[Array, " N 3"] = jnp.asarray(pcl.points, jnp.float32)
        output_jax: Bool[Array, " N"] = bounds_contains(self.bounds, points_jax)
        points: wp.array = wp.from_jax(points_jax[output_jax], dtype=wp.vec3f)
        output: wp.array = wp.zeros(points.shape, dtype=wp.bool)
        wp.launch(
            _contains_kernel,
            dim=points.shape,
            inputs=[self.mesh_wp.id, points, self.scale],
            outputs=[output],
        )
        output_jax = output_jax.at[output_jax].set(wp.to_jax(output))
        return output_jax


def contains(mesh: Any, pcl: Any) -> Bool[Array, " N"]:
    solver = MeshContainsPoints(mesh)
    return solver.contains(pcl)


@wp.kernel
@no_type_check
def _contains_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    # outputs
    output: wp.array(dtype=wp.bool),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    output[tid] = query.sign < 0
