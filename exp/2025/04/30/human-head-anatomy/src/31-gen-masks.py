from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")
    skin: Path = cherries.input("12-skin.vtp")
    tetmesh: Path = cherries.input("30-muscle-fraction.vtu")

    output: Path = cherries.output("31-masks.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    skin: pv.PolyData = melon.load_polydata(cfg.skin)

    cranium: pv.PolyData = melon.tri.extract_groups(full, groups["Cranium"])
    mandible: pv.PolyData = melon.tri.extract_groups(full, groups["Mandible"])

    cranium.cell_data["IsCranium"] = np.ones((cranium.n_cells,), np.bool)
    mandible.cell_data["IsCranium"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["IsCranium"] = np.zeros((skin.n_cells,), np.bool)

    cranium.cell_data["IsFace"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["IsFace"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["IsFace"] = melon.tri.select_groups(
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

    cranium.cell_data["IsLipTop"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["IsLipTop"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["IsLipTop"] = melon.tri.select_groups(
        skin, ["LipTop", "LipInnerTop", "MouthSocketTop"]
    )

    cranium.cell_data["IsLipBottom"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["IsLipBottom"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["IsLipBottom"] = melon.tri.select_groups(
        skin, ["LipBottom", "LipInnerBottom", "MouthSocketBottom"]
    )

    cranium.cell_data["IsMandible"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["IsMandible"] = np.ones((mandible.n_cells,), np.bool)
    skin.cell_data["IsMandible"] = np.zeros((skin.n_cells,), np.bool)

    cranium.cell_data["IsSkin"] = np.zeros((cranium.n_cells,), np.bool)
    mandible.cell_data["IsSkin"] = np.zeros((mandible.n_cells,), np.bool)
    skin.cell_data["IsSkin"] = np.ones((skin.n_cells,), np.bool)

    data_names: list[str] = [
        "IsCranium",
        "IsFace",
        "IsLipBottom",
        "IsLipTop",
        "IsMandible",
        "IsSkin",
    ]

    mesh.point_data["_PointId"] = np.arange(mesh.n_points)
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface = melon.transfer_tri_cell_to_point_category(
        pv.merge([cranium, mandible, skin]),
        surface,
        data=data_names,
        fill=False,
        nearest=melon.NearestPointOnSurface(
            distance_threshold=0.01, normal_threshold=None
        ),
    )

    # fix skull mask around mouth
    orbicularis_oris: pv.PolyData = melon.tri.extract_groups(
        full, "Orbicularis_oris001"
    )
    orbicularis_oris_mask: Bool[Array, " p"] = melon.bounds_contains(
        orbicularis_oris.bounds, surface.points
    )
    surface.point_data["IsCranium"][orbicularis_oris_mask] = False  # pyright: ignore[reportArgumentType]
    surface.point_data["IsMandible"][orbicularis_oris_mask] = False  # pyright: ignore[reportArgumentType]
    melon.save(cherries.temp("21-surface.vtp"), surface)

    mesh = melon.transfer_tri_point_to_tet(
        surface, mesh, data=data_names, fill=False, point_id="_PointId"
    )
    del mesh.point_data["_PointId"]
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
