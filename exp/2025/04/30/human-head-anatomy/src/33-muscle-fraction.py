from pathlib import Path

import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    n_samples: int = 100

    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    tetgen: Path = cherries.data("02-intermediate/20-tetgen.vtu")


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
        ).principal_inertia_vectors[:, 0]
        ic(muscle.field_data["muscle-direction"])

    for cid, cell in grapes.track(
        enumerate(tetmesh.cell),
        total=tetmesh.n_cells,
        callback_stop=grapes.timing.callback.NOOP,
    ):
        cell: pv.Cell
        barycentric: Float[np.ndarray, "N 3"] = melon.sample_barycentric_coords(
            (cfg.n_samples, 4)
        )
        samples: Float[np.ndarray, "N 3"] = melon.barycentric_to_points(
            einops.repeat(cell.points, "B D -> N B D", N=cfg.n_samples), barycentric
        )
        is_in: Bool[np.ndarray, " N"] = np.zeros((cfg.n_samples,), dtype=bool)
        for muscle in muscles:
            is_in |= melon.triangle.contains(muscle, samples)
            tetmesh.cell_data["muscle-direction"][cid] = muscle.field_data[
                "muscle-direction"
            ]
        tetmesh.cell_data["muscle-fraction"][cid] += (
            np.count_nonzero(is_in) / cfg.n_samples
        )


if __name__ == "__main__":
    cherries.run(main)
