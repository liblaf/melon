import itertools

import einops
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Integer


def cell_neighbors(cells: Integer[ArrayLike, "C V"]) -> Integer[Array, "N 2"]:
    cells: Integer[Array, "C V"] = jnp.asarray(cells, dtype=jnp.int32)
    n_cells: int = cells.shape[0]
    combinations: Integer[Array, "4 3"] = jnp.asarray(
        list(itertools.combinations(range(4), 3))
    )
    faces: Integer[Array, "C 4 3"] = cells[:, combinations]
    faces = jnp.sort(faces, axis=-1)
    cell_idx: Integer[Array, " C*4"] = einops.repeat(
        jnp.arange(n_cells), "C -> (C 4)", C=n_cells
    )
    faces: Integer[Array, "C*4 3"] = einops.rearrange(faces, "C F V -> (C F) V")
    order: Integer[Array, " C*4"] = jnp.lexsort(faces.T)
    faces_sorted: Integer[Array, " C*4 3"] = faces[order]
    cell_idx_sorted: Integer[Array, " C*4"] = cell_idx[order]
    mask: Array = jnp.all(faces_sorted[:-1] == faces_sorted[1:], axis=-1)
    neighbors: Integer[Array, "N 2"] = jnp.stack(
        [cell_idx_sorted[:-1][mask], cell_idx_sorted[1:][mask]], axis=-1
    )
    neighbors = jnp.sort(neighbors, axis=-1)
    neighbors = jnp.unique(neighbors, axis=0)
    return neighbors
