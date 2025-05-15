from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

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

    for muscle in muscles:
        components: Float[np.ndarray, " 3"] = melon.as_trimesh(
            muscle
        ).principal_inertia_components
        vectors: Float[np.ndarray, "3 3"] = melon.as_trimesh(
            muscle
        ).principal_inertia_vectors
        index: Integer[np.ndarray, " 3"] = np.argsort(components)
        muscle.field_data["moment-inertia"] = melon.as_trimesh(muscle).moment_inertia
        muscle.field_data["principal-inertia-vectors"] = vectors[index]
        muscle.field_data["principal-inertia-components"] = components[index]
        ic(
            muscle.cell_data["GroupIds"][0],
            muscle.field_data["GroupNames"][muscle.cell_data["GroupIds"][0]],
        )

    pv.set_plot_theme("document_pro")
    plotter = pv.Plotter()
    for muscle in muscles:
        plotter.add_mesh(muscle)
        plotter.add_arrows(
            muscle.center_of_mass(),
            muscle.field_data["principal-inertia-vectors"][0],
            mag=2,
            color="red",
        )
        plotter.add_arrows(
            muscle.center_of_mass(),
            muscle.field_data["principal-inertia-vectors"][1],
            mag=1,
            color="green",
        )
    # plotter.show()


if __name__ == "__main__":
    cherries.run(main)
