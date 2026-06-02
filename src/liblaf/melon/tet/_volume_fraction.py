import math
from typing import no_type_check

import einops
import numpy as np
import pyvista as pv
import scipy
import torch
import warp as wp
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from liblaf.melon import io


def _torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.get_default_device()


def volume_fraction(
    mesh: pv.UnstructuredGrid,
    surface: pv.PolyData,
    *,
    n_samples: int = 1024,
    split_size: int = 100_000_000,
) -> Float[np.ndarray, " c"]:
    with _torch_device():
        surface_wp: wp.Mesh = io.as_warp_mesh(surface)
        cells: Integer[np.ndarray, " c 4"] = mesh.cells_dict[pv.CellType.TETRA]  # ty:ignore[invalid-argument-type]
        cells: Integer[Tensor, "c 4"] = torch.tensor(cells)
        points: Float[Tensor, "p 3"] = torch.tensor(mesh.points, dtype=torch.float32)
        contains: Bool[Tensor, " p"] = _contains_points(surface_wp, points)

        fraction: Float[Tensor, " c"] = torch.zeros((mesh.n_cells,))
        inside: Bool[Tensor, " c"] = torch.all(contains[cells], dim=-1)
        outside: Bool[Tensor, " c"] = ~torch.any(contains[cells], dim=-1)
        fraction[inside] = 1.0
        fraction[outside] = 0.0

        remainder: Integer[Tensor, " r"] = torch.nonzero(~(inside | outside)).squeeze()
        barycentric: Float[Tensor, "s 4"] = _sample_barycentric_coordinates(n_samples)
        n_samples: int = barycentric.shape[0]
        for chunk in torch.split(remainder, split_size // n_samples):
            samples: Float[Tensor, "k s 3"] = _barycentric_to_points(
                points, cells[chunk], barycentric
            )
            samples: Float[Tensor, "k*s 3"] = samples.reshape(-1, 3)
            contains: Bool[Tensor, " k*s"] = _contains_points(surface_wp, samples)
            contains: Bool[Tensor, "k s"] = contains.reshape(-1, n_samples)
            fraction[chunk] = torch.mean(contains.float(), dim=-1)

        return fraction.numpy(force=True)


def _sample_barycentric_coordinates(n: int) -> Float[Tensor, "s 4"]:
    m: int = math.ceil(math.log2(n))
    sobol: scipy.stats.qmc.Sobol = scipy.stats.qmc.Sobol(d=3, scramble=False)
    samples: Float[np.ndarray, "s 3"] = sobol.random_base2(m)
    n_samples: int = samples.shape[0]
    samples: Float[Tensor, "s 3"] = torch.tensor(samples, dtype=torch.float32)
    samples, _indices = torch.sort(samples)
    prepend: Float[Tensor, "s 1"] = torch.zeros((n_samples, 1))
    append: Float[Tensor, "s 1"] = torch.ones((n_samples, 1))
    samples: Float[Tensor, "s 4"] = torch.diff(samples, prepend=prepend, append=append)
    return samples


def _barycentric_to_points(
    points: Float[Tensor, "p 3"],
    cells: Integer[Tensor, "c 4"],
    barycentric: Float[Tensor, "s 4"],
) -> Float[Tensor, "c s 3"]:
    return einops.einsum(points[cells], barycentric, "c i j, s i -> c s j")


def _contains_points(mesh: wp.Mesh, points: Float[Tensor, "p 3"]) -> Bool[Tensor, " p"]:
    direction: wp.vec3f = wp.vec3(1.0, 0.0, 0.0)
    output: Integer[Tensor, " p"] = torch.empty((points.shape[0],), dtype=torch.int32)
    torch_stream: torch.Stream = torch.cuda.current_stream()
    warp_stream: wp.Stream = wp.stream_from_torch(torch_stream)
    with wp.ScopedStream(warp_stream):
        wp.launch(
            _mesh_query_ray,
            dim=points.shape[0],
            inputs=[mesh.id, points, direction],
            outputs=[output],
        )
    return output % 2 == 1


@no_type_check
@wp.kernel
def _mesh_query_ray(
    mesh_id: wp.uint64,
    points: wp.array1d[wp.vec3f],
    direction: wp.vec3f,
    output: wp.array1d[wp.int32],
) -> None:
    tid = wp.tid()
    output[tid] = wp.mesh_query_ray_count_intersections(mesh_id, points[tid], direction)
