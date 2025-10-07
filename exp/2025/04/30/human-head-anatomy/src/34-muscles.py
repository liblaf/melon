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

    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")
    tetgen: Path = cherries.input("02-intermediate/23-tetgen.vtu")

    output: Path = cherries.output("02-intermediate/34-tetgen.vtu")


@attrs.define
class TaskResult:
    cid: int
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
    is_in: Bool[np.ndarray, " N"] = np.zeros((n_samples,), dtype=bool)
    muscle_orientation: Float[np.ndarray, "3 3"] = np.zeros((3, 3))
    muscle_fraction: float = 0.0
    major_muscle_fraction: float = 0.0
    for muscle in muscles:
        contains: Bool[np.ndarray, " N"] = melon.tri.contains(muscle, samples)
        n_contains: int = np.count_nonzero(contains)  # pyright: ignore[reportAssignmentType]
        if n_contains == 0:
            continue
        major_muscle_fraction = max(major_muscle_fraction, n_contains / n_samples)
        is_in |= contains
        muscle_orientation = muscle.field_data["muscle-orientation"]
        assert muscle_orientation.shape == (3, 3)
    muscle_fraction = np.count_nonzero(is_in) / n_samples  # pyright: ignore[reportAssignmentType]

    return TaskResult(
        cid=cid, muscle_fraction=muscle_fraction, muscle_orientation=muscle_orientation
    )


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)

    # groups: list[str] = grapes.load(cfg.groups)["Muscles"]
    groups: list[str] = [
        "Levator_labii_superioris001",
        "Zygomaticus_major001",
        "Zygomaticus_minor001",
    ]
    muscles: list[pv.PolyData] = []
    for group in groups:
        muscle: pv.PolyData = melon.tri.extract_groups(full, group)
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
        vectors = vectors[np.argsort(components)]
        muscle.field_data["muscle-orientation"] = vectors.ravel()

    tetmesh.cell_data["muscle-fraction"] = np.zeros((tetmesh.n_cells,))
    tetmesh.cell_data["muscle-orientation"] = np.zeros((tetmesh.n_cells, 3, 3))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in grapes.track(
            executor.map(
                process_muscle,
                itertools.repeat(tetmesh),
                itertools.repeat(muscles),
                range(tetmesh.n_cells),
                itertools.repeat(cfg.n_samples),
                chunksize=tetmesh.n_cells // 256,
            ),
            total=tetmesh.n_cells,
        ):
            tetmesh.cell_data["muscle-fraction"][result.cid] = result.muscle_fraction
            tetmesh.cell_data["muscle-orientation"][result.cid] = (
                result.muscle_orientation.ravel()
            )
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
        muscle: pv.PolyData
        contains: Bool[np.ndarray, " N"] = melon.tri.contains(muscle, samples)
        n_contains: int = np.count_nonzero(contains)  # pyright: ignore[reportAssignmentType]
        if n_contains == 0:
            continue
        if n_contains / n_samples > major_muscle_fraction:
            major_muscle_fraction = n_contains / n_samples
            major_muscle = muscle
        is_in |= contains
        muscle_direction = muscle.field_data["muscle-direction"]
    muscle_fraction = np.count_nonzero(is_in) / n_samples  # pyright: ignore[reportAssignmentType]

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
