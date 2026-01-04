import math

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Bool, Float, Integer

from liblaf.melon.typing import TetMeshLike, TriMeshLike


def compute_volume_fraction(
    mesh: TetMeshLike,
    surface: TriMeshLike,
    *,
    chunk_size: int = 10**6,
    n_samples: int = 1000,
) -> Float[Array, " cells"]:
    from liblaf.melon import io, tri
    from liblaf.melon.barycentric import (
        barycentric_to_points,
        sample_barycentric_coords,
    )

    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    surface: pv.PolyData = io.as_polydata(surface)
    points: Float[Array, "points 3"] = jnp.asarray(mesh.points)
    cells: Integer[Array, "cells 4"] = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA])  # pyright: ignore[reportArgumentType]
    volume_fraction: Float[Array, " cells"] = jnp.zeros((mesh.n_cells,))
    query = tri.MeshQuery(surface)

    # 1. AABB check
    # check if the AABB of the tetra intersects the AABB of the surface
    cell_points: Float[Array, "cells 4 3"] = points[cells]
    cell_bound_min: Float[Array, "cells 3"] = jnp.min(cell_points, axis=1)
    cell_bound_max: Float[Array, "cells 3"] = jnp.max(cell_points, axis=1)
    in_aabb: Bool[Array, " cells"] = jnp.all(
        (query.bounds[0] <= cell_bound_max) & (cell_bound_min <= query.bounds[1]),
        axis=-1,
    )

    # 2. SDF check
    # use Signed Distance Function to classify tetras as fully inside, fully outside, or crossing
    candidates: Integer[Array, " candidates"] = jnp.flatnonzero(in_aabb)
    if candidates.size == 0:
        return volume_fraction
    candidate_cell_points: Float[Array, "candidates 4 3"] = cell_points[candidates]
    centroids: Float[Array, "candidates 3"] = jnp.mean(candidate_cell_points, axis=1)
    # maximum distance from centroid to vertices
    radius: Float[Array, " candidates"] = jnp.max(
        jnp.linalg.norm(candidate_cell_points - centroids[:, jnp.newaxis, :], axis=-1),
        axis=-1,
    )
    sdf: Float[Array, " candidates"] = query.signed_distance(centroids)
    fully_inside: Bool[Array, " candidates"] = sdf < -radius
    fully_outside: Bool[Array, " candidates"] = sdf > radius
    volume_fraction = volume_fraction.at[candidates[fully_inside]].set(1.0)

    # 3. sampling
    candidates: Integer[Array, " candidates"] = candidates[
        ~(fully_inside | fully_outside)
    ]
    if candidates.size == 0:
        return volume_fraction
    for chunk in jnp.array_split(
        candidates, math.ceil(candidates.size * n_samples / chunk_size)
    ):
        barycentric: Float[Array, "cells samples 4"] = sample_barycentric_coords(
            (chunk.size, n_samples, 4)
        )
        samples: Float[Array, "cells samples 3"] = barycentric_to_points(
            points[cells[chunk]][:, jnp.newaxis, :, :], barycentric
        )
        contains: Bool[Array, "cells samples"] = query.contains(
            samples.reshape(chunk.size * n_samples, 3)
        ).reshape(chunk.size, n_samples)
        fraction: Float[Array, " cells"] = (
            jnp.count_nonzero(contains, axis=-1) / n_samples
        )
        volume_fraction = volume_fraction.at[chunk].set(fraction)
    return volume_fraction
