import logging
import math
from pathlib import Path

import einops
import numpy as np
import pyvista as pv
import scipy
import torch
import warp as wp
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from liblaf import cherries, melon
from liblaf.melon.tri import MeshQueryPointResult, mesh_query_point

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    muscles: Path = cherries.input("30-muscles.m.vtkhdf")
    smas: Path = cherries.input("32-smas.vtp")


def compute_center_radius(mesh: pv.UnstructuredGrid) -> Float[Tensor, " C"]:
    centers: pv.PolyData = mesh.cell_centers()
    cells: Integer[Tensor, "C 4"] = torch.tensor(
        mesh.cells_dict[pv.CellType.TETRA],  # ty:ignore[invalid-argument-type]
        dtype=torch.int32,
    )
    center_points: Float[Tensor, "C 3"] = torch.tensor(
        centers.points, dtype=torch.float32
    )
    mesh_points: Float[Tensor, "P 3"] = torch.tensor(mesh.points, dtype=torch.float32)
    centers_radius: Float[Tensor, " C"] = torch.amax(
        torch.norm(center_points[:, torch.newaxis, :] - mesh_points[cells], dim=-1),
        dim=-1,
    )
    return centers_radius


def broad_phase_muscle(
    mesh: pv.UnstructuredGrid, muscles: pv.MultiBlock, smas: pv.PolyData
) -> None:
    muscles: pv.UnstructuredGrid = muscles.combine()
    muscles: pv.PolyData = muscles.extract_surface(algorithm=None)
    muscles_wp: wp.Mesh = melon.io.as_warp_mesh(muscles)
    face_muscle_id: Integer[Tensor, " F"] = torch.tensor(
        muscles.cell_data["MuscleId"], dtype=torch.int32
    )

    centers: pv.PolyData = mesh.cell_centers()
    centers: Float[Tensor, "C 3"] = torch.tensor(centers.points, dtype=torch.float32)
    center_radius: Float[Tensor, " C"] = compute_center_radius(mesh)

    query_muscle: MeshQueryPointResult = mesh_query_point(
        muscles_wp, centers, max_dist=muscles.length
    )
    assert torch.all(query_muscle.result)
    in_muscle: Bool[Tensor, " C"] = query_muscle.distance < -center_radius
    out_muscle: Bool[Tensor, " C"] = query_muscle.distance > center_radius
    muscle_fraction: Float[Tensor, " C"] = torch.full((mesh.n_cells,), torch.nan)
    muscle_fraction[in_muscle] = 1.0
    muscle_fraction[out_muscle] = 0.0
    muscle_id: Integer[Tensor, " C"] = torch.full(
        (mesh.n_cells,), -1, dtype=torch.int32
    )
    muscle_id[in_muscle] = face_muscle_id[query_muscle.face[in_muscle]]

    mesh.cell_data["MuscleFraction"] = muscle_fraction.numpy(force=True)
    mesh.cell_data["MuscleId"] = muscle_id.numpy(force=True)

    smas_wp: wp.Mesh = melon.io.as_warp_mesh(smas)
    query_smas: MeshQueryPointResult = mesh_query_point(
        smas_wp, centers, max_dist=smas.length
    )
    assert torch.all(query_smas.result)
    in_smas: Bool[Tensor, " C"] = query_smas.distance < -center_radius
    out_smas: Bool[Tensor, " C"] = query_smas.distance > center_radius
    smas_fraction: Float[Tensor, " C"] = torch.full((mesh.n_cells,), torch.nan)
    smas_fraction[in_smas] = 1.0
    smas_fraction[out_smas] = 0.0
    aponeurosis_fraction: Float[Tensor, " C"] = torch.where(
        in_muscle, 0.0, smas_fraction
    )

    mesh.cell_data["AponeurosisFraction"] = aponeurosis_fraction.numpy(force=True)
    mesh.cell_data["SmasFraction"] = smas_fraction.numpy(force=True)


def sample_barycentric_coordinates(n: int) -> Float[Tensor, "s 4"]:
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


def barycentric_to_points(
    points: Float[Tensor, "p 3"],
    cells: Integer[Tensor, "c 4"],
    barycentric: Float[Tensor, "s 4"],
) -> Float[Tensor, "c s 3"]:
    return einops.einsum(points[cells], barycentric, "c i j, s i -> c s j")


def filter_mode(x: Integer[Tensor, "K S"]) -> Integer[Tensor, " K"]:
    with x.device:
        n_classes: int = int(x.max().item()) + 1
        if n_classes == 0:
            return torch.full((x.shape[0],), -1, dtype=torch.int32)
        counts: Integer[Tensor, "K N"] = torch.zeros(
            (x.shape[0], n_classes), dtype=torch.int32
        )
        counts.scatter_add_(1, x.clamp_min(0), (x >= 0).int())
        counts_max, modes = torch.max(counts, dim=-1)
        return torch.where(counts_max > 0, modes.int(), -1)


def narrow_phase(
    mesh: pv.UnstructuredGrid, muscles: pv.MultiBlock, smas: pv.PolyData
) -> None:
    muscles: pv.UnstructuredGrid = muscles.combine()
    muscles: pv.PolyData = muscles.extract_surface(algorithm=None)
    face_muscle_id: Integer[Tensor, " F"] = torch.tensor(
        muscles.cell_data["MuscleId"], dtype=torch.int32
    )
    muscles_wp: wp.Mesh = melon.io.as_warp_mesh(muscles)
    smas_wp: wp.Mesh = melon.io.as_warp_mesh(smas)
    aponeurosis_fraction: Float[Tensor, " C"] = torch.tensor(
        mesh.cell_data["AponeurosisFraction"], dtype=torch.float32
    )
    muscle_fraction: Float[Tensor, " C"] = torch.tensor(
        mesh.cell_data["MuscleFraction"], dtype=torch.float32
    )
    muscle_id: Integer[Tensor, " C"] = torch.tensor(
        mesh.cell_data["MuscleId"], dtype=torch.int32
    )
    smas_fraction: Float[Tensor, " C"] = torch.tensor(
        mesh.cell_data["SmasFraction"], dtype=torch.float32
    )
    remainder_mask: Bool[Tensor, " C"] = (
        torch.isnan(aponeurosis_fraction)
        | torch.isnan(muscle_fraction)
        | torch.isnan(smas_fraction)
    )
    remainder: Integer[Tensor, " R"] = torch.nonzero(remainder_mask).squeeze(dim=-1)

    points: Float[Tensor, "P 3"] = torch.tensor(mesh.points, dtype=torch.float32)
    cells: Integer[Tensor, "C 4"] = torch.tensor(
        mesh.cells_dict[pv.CellType.TETRA],  # ty:ignore[invalid-argument-type]
        dtype=torch.int32,
    )
    barycentric: Float[Tensor, "s 4"] = sample_barycentric_coordinates(1000)
    n_samples: int = barycentric.shape[0]

    n_done: int = 0
    logger.debug("narrow phase: %d / %d", n_done, remainder.shape[0])
    for chunk in torch.split(remainder, 50_000_000 // n_samples):
        samples: Float[Tensor, "k s 3"] = barycentric_to_points(
            points, cells[chunk], barycentric
        )
        samples: Float[Tensor, "k*s 3"] = samples.reshape(-1, 3)
        query_muscle: MeshQueryPointResult = mesh_query_point(
            muscles_wp, samples, max_dist=muscles.length
        )
        query_smas: MeshQueryPointResult = mesh_query_point(
            smas_wp, samples, max_dist=smas.length
        )
        assert torch.all(query_muscle.result)
        assert torch.all(query_smas.result)
        in_muscle: Bool[Tensor, " k*s"] = query_muscle.distance < 0
        in_muscle: Bool[Tensor, "k s"] = in_muscle.reshape(-1, n_samples)
        muscle_fraction[chunk] = torch.mean(in_muscle.float(), dim=-1)
        query_muscle_face: Integer[Tensor, "k s"] = query_muscle.face.reshape(
            -1, n_samples
        )
        sample_muscle_id: Integer[Tensor, "k s"] = torch.where(
            in_muscle,
            face_muscle_id[query_muscle_face],
            torch.full_like(query_muscle_face, -1),
        )
        muscle_id[chunk] = filter_mode(sample_muscle_id)

        in_smas: Bool[Tensor, " k*s"] = query_smas.distance < 0
        in_smas: Bool[Tensor, "k s"] = in_smas.reshape(-1, n_samples)
        smas_fraction[chunk] = torch.mean(in_smas.float(), dim=-1)

        in_aponeurosis: Bool[Tensor, " C"] = in_smas & ~in_muscle
        aponeurosis_fraction[chunk] = torch.mean(in_aponeurosis.float(), dim=-1)

        n_done += chunk.shape[0]
        logger.debug("narrow phase: %d / %d", n_done, remainder.shape[0])

    mesh.cell_data["AponeurosisFraction"] = aponeurosis_fraction.numpy(force=True)
    mesh.cell_data["MuscleFraction"] = muscle_fraction.numpy(force=True)
    mesh.cell_data["MuscleId"] = muscle_id.numpy(force=True)
    mesh.cell_data["SmasFraction"] = smas_fraction.numpy(force=True)


def main(cfg: Config) -> None:
    torch.set_default_device("cuda")
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.muscles)
    smas: pv.PolyData = melon.io.load_polydata(cfg.smas)
    mesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        cherries.input(f"40-tetwild-{cfg.suffix}.vtu")
    )

    for muscle_id, muscle in enumerate(muscles):
        muscle.cell_data["MuscleId"] = np.full(
            (muscle.n_cells,), muscle_id, dtype=np.int32
        )
    broad_phase_muscle(mesh, muscles, smas)
    narrow_phase(mesh, muscles, smas)
    mesh.field_data["MuscleName"] = muscles.keys()

    melon.save(mesh, cherries.output(f"41-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
