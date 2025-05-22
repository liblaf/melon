import concurrent
import concurrent.futures
import itertools
from pathlib import Path

import attrs
import einops
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    n_samples: int = 100

    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    tetgen: Path = cherries.data("02-intermediate/23-tetgen.vtu")

    output: Path = cherries.data("02-intermediate/33-muscle-fraction.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)

    groups: list[str] = ["Levator_labii_superioris001"]
    muscles: list[pv.PolyData] = []
    for group in groups:
        muscle: pv.PolyData = melon.triangle.extract_groups(full, group)
        blocks: pv.MultiBlock = muscle.split_bodies(label=True).as_polydata_blocks()
        for block in blocks:
            fixed: pv.PolyData = melon.mesh_fix(block)
            fixed.user_dict["name"] = f"{group}.{block.point_data['RegionId'][0]}"
            ic(fixed.user_dict["name"])
            muscles.append(fixed)

    for muscle in muscles:
        muscle_tm: tm.Trimesh = melon.as_trimesh(muscle)
        vectors: Float[np.ndarray, "3 3"] = muscle_tm.principal_inertia_vectors
        components: Float[np.ndarray, " 3"] = muscle_tm.principal_inertia_components
        index: Integer[np.ndarray, " 3"] = np.argsort(components)
        muscle.field_data["muscle-direction"] = vectors[index[0]]

    tetmesh.cell_data["muscle-direction"] = np.zeros((tetmesh.n_cells, 3))
    tetmesh.cell_data["muscle-fraction"] = np.zeros((tetmesh.n_cells,))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in grapes.track(
            executor.map(
                compute_muscle_fraction,
                itertools.repeat(tetmesh),
                itertools.repeat(muscles),
                itertools.repeat(cfg.n_samples),
                range(tetmesh.n_cells),
                chunksize=tetmesh.n_cells // 256,
            ),
            total=tetmesh.n_cells,
            callback_stop=grapes.nop,
        ):
            tetmesh.cell_data["muscle-direction"][result.cid] = result.muscle_direction
            tetmesh.cell_data["muscle-fraction"][result.cid] = result.muscle_fraction
            tetmesh.cell_data["muscle-name"][result.cid] = result.major_muscle

    melon.save(cfg.output, tetmesh)


@attrs.frozen(kw_only=True)
class Result:
    cid: int
    major_muscle: str | None = None
    muscle_direction: Float[np.ndarray, " 3"]
    muscle_fraction: float


def compute_muscle_fraction(
    tetmesh: pv.UnstructuredGrid, muscles: pv.MultiBlock, n_samples: int, cid: int
) -> Result:
    cell: pv.Cell = tetmesh.get_cell(cid)
    barycentric: Float[np.ndarray, "N 3"] = melon.sample_barycentric_coords(
        (n_samples, 4)
    )
    samples: Float[np.ndarray, "N 3"] = melon.barycentric_to_points(
        einops.repeat(cell.points, "B D -> N B D", N=n_samples), barycentric
    )
    is_in: Bool[np.ndarray, " N"] = np.zeros((n_samples,), dtype=bool)
    muscle_direction: Float[np.ndarray, " 3"] = np.zeros((3,))
    muscle_fraction: float = 0.0
    major_muscle: pv.PolyData | None = None
    major_muscle_fraction: float = 0.0
    for muscle in muscles:
        contains: Bool[np.ndarray, " N"] = melon.triangle.contains(muscle, samples)
        n_contains: int = np.count_nonzero(contains)
        if n_contains == 0:
            continue
        if n_contains / n_samples > major_muscle_fraction:
            major_muscle_fraction = n_contains / n_samples
            major_muscle = muscle
        is_in |= contains
        muscle_direction = muscle.field_data["muscle-direction"]
    muscle_fraction = np.count_nonzero(is_in) / n_samples

    return Result(
        cid=cid,
        major_muscle=major_muscle.user_dict["name"]
        if major_muscle is not None
        else None,
        muscle_direction=muscle_direction,
        muscle_fraction=muscle_fraction,
    )


if __name__ == "__main__":
    cherries.run(main)
