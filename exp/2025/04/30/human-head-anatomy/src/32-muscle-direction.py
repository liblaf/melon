from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    muscles_union: pv.PolyData = melon.tri.extract_groups(full, groups["Muscles"])
    blocks: pv.MultiBlock = muscles_union.split_bodies().as_polydata_blocks()
    for muscle in blocks:
        muscle: pv.PolyData
        muscle.user_dict["name"] = muscle.field_data["GroupNames"][
            muscle.cell_data["GroupIds"][0]
        ]
    muscles: list[pv.PolyData] = [melon.mesh_fix(muscle) for muscle in blocks]

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
        ic(muscle.user_dict["name"])

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
    plotter.show()


if __name__ == "__main__":
    cherries.run(main, profile="playground")
