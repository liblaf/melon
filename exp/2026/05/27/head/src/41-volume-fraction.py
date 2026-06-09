import logging
from pathlib import Path

import attrs
import einops
import liblaf.logging
import numpy as np
import pyvista as pv
import torch
import warp as wp
from jaxtyping import Bool, Float, Integer
from rich.progress import Progress, TaskID
from torch import Tensor

from liblaf import cherries, melon
from liblaf.melon.tet import TetraCenterRadiusResult, tetra_center_radius

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    muscles: Path = cherries.input("30-muscles.m.vtkhdf")
    smas: Path = cherries.input("32-smas.vtp")


@attrs.define
class FractionResult:
    aponeurosis_fraction: Float[Tensor, " c"]
    muscle_fraction: Float[Tensor, " c"]
    muscle_id: Integer[Tensor, " c"]
    smas_fraction: Float[Tensor, " c"]


def broad_phase(
    center_radius: TetraCenterRadiusResult,
    muscles: list[wp.Mesh],
    muscles_union: wp.Mesh,
    smas: wp.Mesh,
) -> FractionResult:
    n_cells: int = center_radius.center.shape[0]

    dist_to_muscle: Bool[Tensor, " C"] = melon.tri.implicit_distance(
        muscles_union, center_radius.center
    )
    in_muscle: Bool[Tensor, " C"] = dist_to_muscle < -center_radius.radius
    out_muscle: Bool[Tensor, " C"] = dist_to_muscle > center_radius.radius

    dist_to_smas: Bool[Tensor, " C"] = melon.tri.implicit_distance(
        smas, center_radius.center
    )
    in_smas: Bool[Tensor, " C"] = dist_to_smas < -center_radius.radius
    out_smas: Bool[Tensor, " C"] = dist_to_smas > center_radius.radius

    muscle_fraction: Float[Tensor, " C"] = torch.full((n_cells,), torch.nan)
    muscle_fraction[in_muscle] = 1.0
    muscle_fraction[out_muscle] = 0.0

    smas_fraction: Float[Tensor, " C"] = torch.full((n_cells,), torch.nan)
    smas_fraction[in_smas] = 1.0
    smas_fraction[out_smas] = 0.0

    aponeurosis_fraction: Float[Tensor, " C"] = torch.full((n_cells,), torch.nan)
    aponeurosis_fraction[in_muscle | ~in_smas] = 0.0
    aponeurosis_fraction[in_smas & ~in_muscle] = 1.0

    muscle_id: Integer[Tensor, " C"] = torch.full((n_cells,), -1, dtype=torch.int32)
    for this_muscle_id, this_muscle in enumerate(muscles):
        dist_to_this_muscle: Bool[Tensor, " C"] = melon.tri.implicit_distance(
            this_muscle, center_radius.center
        )
        in_this_muscle: Bool[Tensor, " C"] = dist_to_this_muscle < -center_radius.radius
        muscle_id[in_this_muscle] = this_muscle_id

    return FractionResult(
        aponeurosis_fraction=aponeurosis_fraction,
        muscle_fraction=muscle_fraction,
        muscle_id=muscle_id,
        smas_fraction=smas_fraction,
    )


def single_muscle_fraction(
    muscle: wp.Mesh,
    center: Float[Tensor, "c 3"],
    radius: Float[Tensor, " c"],
    samples: Float[Tensor, "c s 3"],
) -> Float[Tensor, " c"]:
    distance: Float[Tensor, " c"] = melon.tri.implicit_distance(muscle, center)
    in_muscle: Bool[Tensor, " c"] = distance < -radius
    out_muscle: Bool[Tensor, " c"] = distance > radius
    fraction: Float[Tensor, " c"] = torch.full((center.shape[0],), torch.nan)
    fraction[in_muscle] = 1.0
    fraction[out_muscle] = 0.0
    missing: Bool[Tensor, " c"] = ~(in_muscle | out_muscle)
    if missing.any():
        missing_in_muscle: Bool[Tensor, "m s"] = melon.tri.contains(
            muscle, samples[missing]
        )
        fraction[missing] = missing_in_muscle.float().mean(dim=-1)
    return fraction


def narrow_phase(
    mesh: pv.UnstructuredGrid,
    center_radius: TetraCenterRadiusResult,
    muscles: list[wp.Mesh],
    muscles_union: wp.Mesh,
    smas: wp.Mesh,
    fractions: FractionResult,
) -> FractionResult:
    aponeurosis_fraction: Float[Tensor, " c"] = fractions.aponeurosis_fraction
    muscle_fraction: Float[Tensor, " c"] = fractions.muscle_fraction
    smas_fraction: Float[Tensor, " c"] = fractions.smas_fraction
    muscle_id: Integer[Tensor, " c"] = fractions.muscle_id

    remainder_mask: Bool[Tensor, " c"] = ~(
        torch.isfinite(fractions.aponeurosis_fraction)
        & torch.isfinite(fractions.muscle_fraction)
        & torch.isfinite(fractions.smas_fraction)
    )
    remainder_idx: Integer[Tensor, " r"] = torch.nonzero(remainder_mask).squeeze(-1)
    logger.info(
        "broad phase: %d / %d cells need narrow phase",
        remainder_idx.shape[0],
        mesh.n_cells,
    )

    points: Float[Tensor, "p 3"] = torch.tensor(mesh.points, dtype=torch.float32)
    cells: Integer[Tensor, "c 4"] = torch.tensor(
        mesh.cells_dict[pv.CellType.TETRA],  # ty:ignore[invalid-argument-type]
        dtype=torch.int32,
    )

    N_SAMPLES: int = 1024  # noqa: N806
    barycentric: Float[np.ndarray, "s 4"] = melon.bary.sample(N_SAMPLES, 4)
    barycentric: Float[Tensor, "s 4"] = torch.tensor(barycentric, dtype=torch.float32)

    progress: Progress = liblaf.logging.get_progress()
    task_id: TaskID = progress.add_task("narrow phase", total=remainder_idx.shape[0])
    for cell_id in torch.split(remainder_idx, 100_000_000 // N_SAMPLES):
        chunk_size: int = cell_id.shape[0]

        samples: Float[Tensor, "k s 3"] = einops.einsum(
            points[cells[cell_id]], barycentric, "k b i, s b -> k s i"
        )
        in_any_muscle: Bool[Tensor, "k s"] = melon.tri.contains(muscles_union, samples)
        in_smas: Bool[Tensor, "k s"] = melon.tri.contains(smas, samples)
        in_aponeurosis: Bool[Tensor, "k s"] = in_smas & ~in_any_muscle
        aponeurosis_fraction[cell_id] = in_aponeurosis.float().mean(dim=-1)
        muscle_fraction[cell_id] = in_any_muscle.float().mean(dim=-1)
        smas_fraction[cell_id] = in_smas.float().mean(dim=-1)

        dominant_muscle_fraction: Float[Tensor, " k"] = torch.zeros((chunk_size,))
        dominant_muscle_id: Integer[Tensor, " k"] = torch.full(
            (chunk_size,), -1, dtype=torch.int32
        )

        center: Float[Tensor, " k"] = center_radius.center[cell_id]
        radius: Float[Tensor, " k"] = center_radius.radius[cell_id]
        for this_muscle_id, this_muscle in enumerate(muscles):
            this_muscle_fraction: Float[Tensor, " k"] = single_muscle_fraction(
                muscle=this_muscle, center=center, radius=radius, samples=samples
            )
            update_mask: Bool[Tensor, " k"] = (
                this_muscle_fraction > dominant_muscle_fraction
            )
            dominant_muscle_fraction[update_mask] = this_muscle_fraction[update_mask]
            dominant_muscle_id[update_mask] = this_muscle_id
        muscle_id[cell_id] = dominant_muscle_id

        progress.advance(task_id, cell_id.shape[0])

    return FractionResult(
        aponeurosis_fraction=aponeurosis_fraction,
        muscle_fraction=muscle_fraction,
        muscle_id=muscle_id,
        smas_fraction=smas_fraction,
    )


def main(cfg: Config) -> None:
    torch.set_default_device("cuda")
    mesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        cherries.input(f"40-tetwild-{cfg.suffix}.vtu")
    )
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.muscles)
    smas: pv.PolyData = melon.io.load_polydata(cfg.smas)

    for muscle_id, muscle in enumerate(muscles):
        muscle.cell_data["MuscleId"] = np.full(
            (muscle.n_cells,), muscle_id, dtype=np.int32
        )
    muscles_wp: list[wp.Mesh] = [melon.io.as_warp_mesh(muscle) for muscle in muscles]
    muscles_union: pv.UnstructuredGrid = muscles.combine()
    muscles_union: pv.PolyData = muscles_union.extract_surface(algorithm=None)
    muscles_union_wp: wp.Mesh = melon.io.as_warp_mesh(muscles_union)
    smas_wp: wp.Mesh = melon.io.as_warp_mesh(smas)

    center_radius: TetraCenterRadiusResult = tetra_center_radius(mesh)
    fractions: FractionResult = broad_phase(
        center_radius=center_radius,
        muscles=muscles_wp,
        muscles_union=muscles_union_wp,
        smas=smas_wp,
    )
    fractions: FractionResult = narrow_phase(
        mesh=mesh,
        center_radius=center_radius,
        muscles=muscles_wp,
        muscles_union=muscles_union_wp,
        smas=smas_wp,
        fractions=fractions,
    )

    mesh.cell_data["AponeurosisFraction"] = fractions.aponeurosis_fraction.numpy(
        force=True
    )
    mesh.cell_data["MuscleFraction"] = fractions.muscle_fraction.numpy(force=True)
    mesh.cell_data["MuscleId"] = fractions.muscle_id.numpy(force=True)
    mesh.cell_data["SMASFraction"] = fractions.smas_fraction.numpy(force=True)
    mesh.field_data["MuscleName"] = muscles.keys()

    melon.save(mesh, cherries.output(f"41-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
