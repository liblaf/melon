from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.data("02-intermediate/groups.toml")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    muscles: pv.PolyData = melon.triangle.extract_groups(full, groups["Muscles"])
    muscles: pv.MultiBlock = muscles.split_bodies().as_polydata_blocks()
    muscles: list[pv.PolyData] = [melon.mesh_fix(muscle) for muscle in muscles]

    plotter = pv.Plotter()
    for muscle in muscles:
        components: Float[np.ndarray, " 3"] = melon.as_trimesh(
            muscle
        ).principal_inertia_components
        vectors: Float[np.ndarray, "3 3"] = melon.as_trimesh(
            muscle
        ).principal_inertia_vectors
        plotter.add_mesh(muscle)
        plotter.add_arrows(muscle.center_of_mass(), vectors[0])
        plotter.show()


if __name__ == "__main__":
    cherries.run(main)
