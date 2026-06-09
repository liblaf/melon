from typing import no_type_check

import einops
import numpy as np
import pyvista as pv
import torch
import warp as wp
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from liblaf.melon import bary
from liblaf.melon.utils import warp_stream_from_torch


def tetra_surface_fraction(
    mesh: pv.UnstructuredGrid, surface: wp.Mesh, *, n_samples: int = 1024
) -> Bool[Tensor, "c s"]:
    points: Float[Tensor, "p 3"] = torch.tensor(mesh.points, dtype=torch.float32)
    cells: Integer[Tensor, "c 4"] = torch.tensor(
        mesh.cells_dict[pv.CellType.TETRA],  # ty:ignore[invalid-argument-type]
        dtype=torch.int32,
    )
    barycentric: Float[np.ndarray, "s 4"] = bary.sample(n_samples, 4)
    barycentric: Float[Tensor, "s 4"] = torch.tensor(barycentric, dtype=torch.float32)
    contains: Bool[Tensor, " c s"] = torch.full((mesh.n_cells, n_samples), torch.nan)
    split_size: int = max(1, 100_000_000 // n_samples)
    for chunk in torch.split(torch.arange(mesh.n_cells), split_size):
        chunk_cells: Integer[Tensor, "k 4"] = cells[chunk]
        samples: Float[Tensor, "k s 3"] = einops.einsum(
            points[chunk_cells], barycentric, "k b i, s b -> k s i"
        )
        samples: Float[Tensor, "k*s 3"] = samples.reshape(-1, 3)
        chunk_contains: Bool[Tensor, " k*s"] = _mesh_query_point(surface, samples)
        chunk_contains: Bool[Tensor, "k s"] = chunk_contains.reshape(-1, n_samples)
        contains[chunk] = chunk_contains
    return contains


def _mesh_query_point(
    mesh: wp.Mesh, points: Float[Tensor, "p 3"], max_dist: float = wp.inf
) -> Bool[Tensor, " p"]:
    n_points: int = points.shape[0]
    points_wp: wp.array = wp.from_torch(points, wp.vec3f, return_ctype=True)
    contains_wp: wp.array = wp.zeros((n_points,), wp.bool)
    wp.launch(
        _mesh_query_point_kernel,
        dim=(n_points,),
        inputs=[mesh.id, points_wp, max_dist],
        outputs=[contains_wp],
        stream=warp_stream_from_torch(),
    )
    contains: Bool[Tensor, " p"] = wp.to_torch(contains_wp)
    return contains


@wp.kernel
@no_type_check
def _mesh_query_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    max_dist: wp.float32,
    contains: wp.array1d[wp.bool],
) -> None:
    tid = wp.tid()  # int
    point = points[tid]  # vec3
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    if query.result:
        contains[tid] = query.sign < 0
    else:
        contains[tid] = False
