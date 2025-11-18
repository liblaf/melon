from pathlib import Path

import numpy as np
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")
    skin: Path = cherries.input("12-skin.vtp")
    tetmesh: Path = cherries.input("20-tetgen.vtu")

    output: Path = cherries.output("21-tetmesh.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    skin: pv.PolyData = melon.load_polydata(cfg.skin)

    cranium: pv.PolyData = melon.tri.extract_groups(full, groups["Cranium"])
    mandible: pv.PolyData = melon.tri.extract_groups(full, groups["Mandible"])

    cranium.cell_data["is-cranium"] = np.ones((cranium.n_cells,), np.bool)
    mandible.cell_data["is-cranium"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["is-cranium"] = np.zeros((skin.n_cells,), np.bool)

    cranium.cell_data["is-face"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["is-face"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["is-face"] = melon.tri.select_groups(
        skin,
        [
            "Ear",
            "EarNeckBack",
            "EarSocket",
            "EyeSocketBottom",
            "EyeSocketTop",
            "HeadBack",
            "LipInnerBottom",
            "LipInnerTop",
            "MouthSocketBottom",
            "MouthSocketTop",
            "NeckBack",
            "NeckFront",
            "Nostril",
        ],
        invert=True,
    )

    cranium.cell_data["is-mandible"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["is-mandible"] = np.ones((mandible.n_cells,), np.bool)
    skin.cell_data["is-mandible"] = np.zeros((skin.n_cells,), np.bool)

    cranium.cell_data["is-skin"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["is-skin"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["is-skin"] = np.ones((skin.n_cells,), np.bool)

    mesh.point_data["PointIds"] = np.arange(mesh.n_points)
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface = melon.transfer_tri_cell_to_point_category(
        pv.merge([cranium, mandible, skin]),
        surface,
        data=["is-cranium", "is-face", "is-mandible", "is-skin"],
        fill=False,
        nearest=melon.NearestPointOnSurface(
            ignore_orientation=True, normal_threshold=-np.inf
        ),
    )
    melon.save(cherries.temp("21-surface.vtp"), surface)

    mesh = melon.transfer_tri_point_to_tet(
        surface,
        mesh,
        data=["is-cranium", "is-face", "is-mandible", "is-skin"],
        fill=False,
        point_id="PointIds",
    )
    del mesh.point_data["PointIds"]
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
