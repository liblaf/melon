import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, ArrayLike, Bool, Float


# `trimesh.bounds.contains` is too slow for large point clouds, so we implement our own
def bounds_contains(
    bounds: Float[ArrayLike, "2 3"] | Float[ArrayLike, " 6"] | pv.BoundsTuple,
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
