from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Array, Bool, Float, Key

from liblaf import melon


def contains_trimesh(mesh: Any, pcl: Any) -> Bool[np.ndarray, " N"]:
    mesh_tm: tm.Trimesh = melon.as_trimesh(mesh)
    pcl: pv.PointSet = melon.as_pointset(pcl)
    return mesh_tm.contains(pcl.points)


def test_contains() -> None:
    mesh: pv.PolyData = pv.examples.download_bunny()  # pyright: ignore[reportAssignmentType]
    mesh = melon.mesh_fix(mesh)
    key: Key = jax.random.key(0)
    bounds: Float[Array, " 6"] = jnp.array(mesh.bounds)
    points: Float[Array, "N 3"] = jax.random.uniform(
        key, (1000, 3), minval=bounds[::2], maxval=bounds[1::2]
    )
    actual: Bool[Array, " N"] = melon.tri.contains(mesh, points)
    desired: Bool[np.ndarray, " N"] = contains_trimesh(mesh, points)
    n_diff: int = jnp.count_nonzero(actual != desired)  # pyright: ignore[reportAssignmentType]
    assert n_diff < 1e-2 * points.shape[0]
