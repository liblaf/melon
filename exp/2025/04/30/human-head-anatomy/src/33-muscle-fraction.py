import concurrent
import concurrent.futures
import itertools
from pathlib import Path

import attrs
import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    n_samples: int = 100

    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    tetgen: Path = cherries.data("02-intermediate/20-tetgen.vtu")

    output: Path = cherries.data("02-intermediate/33-muscle-fraction.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)

    muscles_union: pv.PolyData = melon.triangle.extract_groups(
        full, ["Levator_labii_superioris001"]
    )
    muscles: pv.MultiBlock = muscles_union.split_bodies().as_polydata_blocks()
    for muscle in muscles:
        muscle.user_dict["name"] = muscle.field_data["GroupNames"][
            muscle.cell_data["GroupIds"][0]
        ]
        muscle.field_data["muscle-direction"] = melon.as_trimesh(
            muscle
        ).principal_inertia_vectors[0]
        ic(muscle.field_data["muscle-direction"])

    tetmesh.cell_data["muscle-direction"] = np.zeros((tetmesh.n_cells, 3))
    tetmesh.cell_data["muscle-fraction"] = np.zeros((tetmesh.n_cells,))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(
            compute_muscle_fraction,
            itertools.repeat(tetmesh),
            itertools.repeat(muscles),
            itertools.repeat(cfg.n_samples),
            range(tetmesh.n_cells),
        ):
            tetmesh.cell_data["muscle-direction"][result.cid] = result.muscle_direction
            tetmesh.cell_data["muscle-fraction"][result.cid] = result.muscle_fraction

    melon.save(cfg.output, tetmesh)


@attrs.frozen
class Result:
    cid: int
    muscle_fraction: float
    muscle_direction: Float[np.ndarray, " 3"]


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
    for muscle in muscles:
        is_in |= melon.triangle.contains(muscle, samples)
        muscle_direction = muscle.field_data["muscle-direction"]
    muscle_fraction = np.count_nonzero(is_in) / n_samples

    return Result(
        cid=cid, muscle_fraction=muscle_fraction, muscle_direction=muscle_direction
    )


if __name__ == "__main__":
    cherries.run(main)
