import functools
import itertools
from typing import Any

import einops
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer


def cell_neighbors(mesh_or_cells: Any) -> Integer[Array, "N 2"]:
    cells: Integer[Array, "C V"] = _as_cells(mesh_or_cells)
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


@functools.singledispatch
def _as_cells(
    mesh_or_cells: Any,
) -> Integer[Array, "C V"]:
    return jnp.asarray(mesh_or_cells, dtype=jnp.int32)


@_as_cells.register(pv.UnstructuredGrid)
def _as_cells_from_pyvista(
    mesh: pv.UnstructuredGrid,
) -> Integer[Array, "C V"]:
    return _as_cells(mesh.cells_dict[pv.CellType.TETRA])  # pyright: ignore[reportArgumentType]
