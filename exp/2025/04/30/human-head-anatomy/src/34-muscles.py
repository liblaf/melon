import concurrent
import concurrent.futures
import itertools
import multiprocessing
from pathlib import Path

import attrs
import einops
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    n_samples: int = 100

    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")
    tetgen: Path = cherries.input("02-intermediate/23-tetgen.vtu")

    output: Path = cherries.output("02-intermediate/34-tetgen.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)

    groups: list[str] = grapes.load(cfg.groups)["Muscles"]
    # groups: list[str] = [
    #     "Levator_labii_superioris001",
    #     "Zygomaticus_main001",
    #     "Zygomaticus_minor001",
    # ]
    muscles: list[pv.PolyData] = []
    for group in groups:
        muscle: pv.PolyData = melon.tri.extract_groups(full, group)
        blocks: pv.MultiBlock = muscle.split_bodies(label=True).as_polydata_blocks()
        for block in blocks:
            block: pv.PolyData
            fixed: pv.PolyData = melon.mesh_fix(block)
            fixed.user_dict["name"] = f"{group}.{block.point_data['RegionId'][0]}"
            ic(fixed.user_dict["name"])
            muscles.append(fixed)
    muscle_names: list[str] = [m.user_dict["name"] for m in muscles]
    muscle_name_to_id: dict[str, int] = {name: i for i, name in enumerate(muscle_names)}

    for muscle in muscles:
        muscle_tm: tm.Trimesh = melon.as_trimesh(muscle)
        vectors: Float[np.ndarray, "3 3"] = muscle_tm.principal_inertia_vectors
        components: Float[np.ndarray, " 3"] = muscle_tm.principal_inertia_components
        vectors = vectors[np.argsort(components)]
        muscle.field_data["muscle-orientation"] = vectors.ravel()

    tetmesh.cell_data["muscle-id"] = np.full((tetmesh.n_cells,), -1, dtype=int)
    tetmesh.cell_data["muscle-fraction"] = np.zeros((tetmesh.n_cells,))
    tetmesh.cell_data["muscle-orientation"] = np.zeros((tetmesh.n_cells, 3, 3))
    with concurrent.futures.ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("fork")
    ) as executor:
        for result in grapes.track(
            executor.map(
                process_muscle,
                itertools.repeat(tetmesh),
                itertools.repeat(muscles),
                range(tetmesh.n_cells),
                itertools.repeat(cfg.n_samples),
                chunksize=1,
            ),
            total=tetmesh.n_cells,
        ):
            tetmesh.cell_data["muscle-id"][result.cid] = (
                muscle_name_to_id[result.main_muscle_name]
                if result.main_muscle_name is not None
                else -1
            )
            tetmesh.cell_data["muscle-fraction"][result.cid] = result.muscle_fraction
            tetmesh.cell_data["muscle-orientation"][result.cid] = (
                result.muscle_orientation.ravel()
            )

    tetmesh.field_data["muscle-name"] = muscle_names
    melon.save(cfg.output, tetmesh)


@attrs.frozen(kw_only=True)
class TaskResult:
    cid: int
    main_muscle_name: str | None
    muscle_fraction: float
    muscle_orientation: Float[np.ndarray, "3 3"]


def process_muscle(
    tetmesh: pv.UnstructuredGrid, muscles: pv.MultiBlock, cid: int, n_samples: int
) -> TaskResult:
    cell: pv.Cell = tetmesh.get_cell(cid)
    barycentric: Float[np.ndarray, "N 3"] = melon.sample_barycentric_coords(
        (n_samples, 4)
    )
    samples: Float[np.ndarray, "N 3"] = melon.barycentric_to_points(
        einops.repeat(cell.points, "B D -> n_samples B D", n_samples=n_samples),
        barycentric,
    )
    samples_pv: pv.PointSet = melon.as_pointset(samples)
    is_in: Bool[np.ndarray, " N"] = np.zeros((n_samples,), dtype=bool)
    muscle_orientation: Float[np.ndarray, "3 3"] = np.zeros((3, 3))
    muscle_fraction: float = 0.0
    main_muscle_fraction: float = 0.0
    main_muscle_name: str | None = None
    for muscle in muscles:
        muscle: pv.PolyData
        contains: Bool[np.ndarray, " N"] = melon.tri.contains(muscle, samples_pv)
        n_contains: int = np.count_nonzero(contains)  # pyright: ignore[reportAssignmentType]
        if n_contains == 0:
            continue
        is_in |= contains
        if n_contains / n_samples > main_muscle_fraction:
            main_muscle_fraction = n_contains / n_samples
            main_muscle_name = muscle.user_dict["name"]
            muscle_orientation = np.reshape(
                muscle.field_data["muscle-orientation"], shape=(3, 3)
            )
    muscle_fraction = np.count_nonzero(is_in) / n_samples  # pyright: ignore[reportAssignmentType]

    return TaskResult(
        cid=cid,
        main_muscle_name=main_muscle_name,
        muscle_fraction=muscle_fraction,
        muscle_orientation=muscle_orientation,
    )


if __name__ == "__main__":
    cherries.main(main)
