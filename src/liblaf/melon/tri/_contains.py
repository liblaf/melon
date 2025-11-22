from typing import Any, no_type_check

import jax
import jax.numpy as jnp
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Array, ArrayLike, Bool, Float

from liblaf.melon import io


# `trimesh.bounds.contains` is too slow for large point clouds, so we implement our own
def bounds_contains(
    bounds: Float[ArrayLike, "2 3"] | Float[ArrayLike, " 6"],
    points: Float[ArrayLike, "N 3"],
) -> Bool[Array, " N"]:
    bounds = jnp.asarray(bounds)
    if bounds.shape == (6,):
        bounds = bounds.reshape(3, 2).T
    points = jnp.asarray(points)
    return _bounds_contains_jit(bounds, points)


@jax.jit
def _bounds_contains_jit(
    bounds: Float[Array, "2 3"], points: Float[Array, "N 3"]
) -> Bool[Array, " N"]:
    return jnp.all(bounds[0] < points, axis=-1) & jnp.all(points < bounds[1], axis=-1)


def contains(mesh: Any, pcl: Any) -> Bool[Array, " N"]:
    mesh_tm: tm.Trimesh = io.as_trimesh(mesh)
    mesh_wp: wp.Mesh = io.as_warp_mesh(mesh)
    pcl: pv.PointSet = io.as_pointset(pcl)
    points_jax: Float[Array, " N 3"] = jnp.asarray(pcl.points, jnp.float32)
    output_jax: Bool[Array, " N"] = bounds_contains(mesh_tm.bounds, points_jax)
    points: wp.array = wp.from_jax(points_jax[output_jax], dtype=wp.vec3f)
    max_dist: float = mesh_tm.scale
    output: wp.array = wp.zeros(points.shape, dtype=wp.bool)
    wp.launch(
        _contains_kernel,
        dim=points.shape,
        inputs=[mesh_wp.id, points, max_dist],
        outputs=[output],
    )
    output_jax = output_jax.at[output_jax].set(wp.to_jax(output))
    return output_jax


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
