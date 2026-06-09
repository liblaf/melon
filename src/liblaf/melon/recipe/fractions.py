import logging

import attrs
import einops
import liblaf.logging
import numpy as np
import pyvista as pv
import torch
import warp as wp
from jaxtyping import Bool, Float, Integer
from liblaf.logging import Progress
from rich.progress import TaskID
from torch import Tensor

from liblaf.melon import bary, io, tri
from liblaf.melon.tet import TetraCenterRadiusResult, tetra_center_radius

logger: logging.Logger = logging.getLogger(__name__)


@attrs.frozen
class FractionResult:
    aponeurosis_fraction: Float[Tensor, " c"]
    muscle_fraction: Float[Tensor, " c"]
    muscle_id: Integer[Tensor, " c"]
    smas_fraction: Float[Tensor, " c"]


def count_to_fraction(
    count: Integer[Tensor, "c s"], dim: int = -1
) -> Float[Tensor, " c"]:
    return count.count_nonzero(dim=dim) / count.shape[dim]


def broad_phase(
    center_radius: TetraCenterRadiusResult,
    muscles: list[wp.Mesh],
    muscles_union: wp.Mesh,
    smas: wp.Mesh,
) -> FractionResult:
    n_cells: int = center_radius.center.shape[0]

    dist_to_muscle: Bool[Tensor, " c"] = tri.implicit_distance(
        muscles_union, center_radius.center
    )
    in_muscle: Bool[Tensor, " c"] = dist_to_muscle < -center_radius.radius
    out_muscle: Bool[Tensor, " c"] = dist_to_muscle > center_radius.radius

    dist_to_smas: Bool[Tensor, " c"] = tri.implicit_distance(smas, center_radius.center)
    in_smas: Bool[Tensor, " c"] = dist_to_smas < -center_radius.radius
    out_smas: Bool[Tensor, " c"] = dist_to_smas > center_radius.radius

    muscle_fraction: Float[Tensor, " c"] = torch.full((n_cells,), torch.nan)
    muscle_fraction[in_muscle] = 1.0
    muscle_fraction[out_muscle] = 0.0

    smas_fraction: Float[Tensor, " c"] = torch.full((n_cells,), torch.nan)
    smas_fraction[in_smas] = 1.0
    smas_fraction[out_smas] = 0.0

    aponeurosis_fraction: Float[Tensor, " c"] = torch.full((n_cells,), torch.nan)
    aponeurosis_fraction[in_muscle | ~in_smas] = 0.0
    aponeurosis_fraction[in_smas & ~in_muscle] = 1.0

    muscle_id: Integer[Tensor, " c"] = torch.full((n_cells,), -1, dtype=torch.int32)
    for this_muscle_id, this_muscle in enumerate(muscles):
        dist_to_this_muscle: Bool[Tensor, " c"] = tri.implicit_distance(
            this_muscle, center_radius.center
        )
        in_this_muscle: Bool[Tensor, " c"] = dist_to_this_muscle < -center_radius.radius
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
    distance: Float[Tensor, " c"] = tri.implicit_distance(muscle, center)
    in_muscle: Bool[Tensor, " c"] = distance < -radius
    out_muscle: Bool[Tensor, " c"] = distance > radius
    fraction: Float[Tensor, " c"] = torch.full((center.shape[0],), torch.nan)
    fraction[in_muscle] = 1.0
    fraction[out_muscle] = 0.0
    missing: Bool[Tensor, " c"] = ~(in_muscle | out_muscle)
    if missing.any():
        missing_in_muscle: Bool[Tensor, "m s"] = tri.contains(muscle, samples[missing])
        fraction[missing] = count_to_fraction(missing_in_muscle)
    return fraction


def narrow_phase(
    mesh: pv.UnstructuredGrid,
    center_radius: TetraCenterRadiusResult,
    muscles: list[wp.Mesh],
    muscles_union: wp.Mesh,
    smas: wp.Mesh,
    fractions: FractionResult,
    *,
    n_samples: int = 1024,
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

    barycentric: Float[np.ndarray, "s 4"] = bary.sample(n_samples, 4)
    barycentric: Float[Tensor, "s 4"] = torch.tensor(barycentric, dtype=torch.float32)

    progress: Progress = liblaf.logging.get_progress()
    task_id: TaskID = progress.add_task("narrow phase", total=remainder_idx.shape[0])
    for cell_id in torch.split(remainder_idx, 100_000_000 // n_samples):
        chunk_size: int = cell_id.shape[0]

        samples: Float[Tensor, "k s 3"] = einops.einsum(
            points[cells[cell_id]], barycentric, "k b i, s b -> k s i"
        )
        in_any_muscle: Bool[Tensor, "k s"] = tri.contains(muscles_union, samples)
        in_smas: Bool[Tensor, "k s"] = tri.contains(smas, samples)
        in_aponeurosis: Bool[Tensor, "k s"] = in_smas & ~in_any_muscle
        aponeurosis_fraction[cell_id] = count_to_fraction(in_aponeurosis)
        muscle_fraction[cell_id] = count_to_fraction(in_any_muscle)
        smas_fraction[cell_id] = count_to_fraction(in_smas)

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


def compute_fractions(
    mesh: pv.UnstructuredGrid,
    muscles: pv.MultiBlock,
    smas: pv.PolyData,
    *,
    n_samples: int = 1024,
) -> FractionResult:
    for muscle_id, muscle in enumerate(muscles):
        muscle.cell_data["MuscleId"] = np.full(
            (muscle.n_cells,), muscle_id, dtype=np.int32
        )
    muscles_wp: list[wp.Mesh] = [io.as_warp_mesh(muscle) for muscle in muscles]
    muscles_union: pv.UnstructuredGrid = muscles.combine()
    muscles_union: pv.PolyData = muscles_union.extract_surface(algorithm=None)
    muscles_union_wp: wp.Mesh = io.as_warp_mesh(muscles_union)
    smas_wp: wp.Mesh = io.as_warp_mesh(smas)

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
        n_samples=n_samples,
    )
    return fractions
