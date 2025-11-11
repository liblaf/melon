from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


def classify(
    surface: pv.PolyData, include: pv.PolyData, exclude: pv.PolyData
) -> Bool[np.ndarray, " N"]:
    nearest_algo: melon.NearestAlgorithm = melon.NearestPointOnSurface(
        distance_threshold=np.inf,
        normal_threshold=-np.inf,
    )
    nearest_include: melon.NearestResult = melon.nearest(
        include, surface, algo=nearest_algo
    )
    nearest_exclude: melon.NearestResult = melon.nearest(
        exclude, surface, algo=nearest_algo
    )
    is_include: Bool[np.ndarray, " N"] = (
        nearest_include.distance <= nearest_exclude.distance
    )
    return is_include


def transfer_trimesh_point_data_to_tetmesh(
    tetmesh: pv.UnstructuredGrid, surface: pv.PolyData, data_name: str
) -> pv.UnstructuredGrid:
    point_ids: Integer[np.ndarray, " N"] = surface.point_data["point-id"]
    data: Bool[np.ndarray, " N"] = np.zeros((tetmesh.n_points,), dtype=bool)
    data[point_ids[surface.point_data[data_name]]] = True
    tetmesh.point_data[data_name] = data
    return tetmesh


class Config(cherries.BaseConfig):
    source: Path = cherries.input("40-expression-flame.vtp")
    tetgen: Path = cherries.input("02-intermediate/34-tetgen.vtu")

    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")
    skin: Path = cherries.input("02-intermediate/12-skin.vtp")

    output: Path = cherries.output("40-expression.vtu")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)
    source: pv.PolyData = melon.load_polydata(cfg.source)
    tetmesh.point_data["point-id"] = np.arange(tetmesh.n_points)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    source = melon.fast_wrapping(source, surface)
    source.compute_normals(auto_orient_normals=True, inplace=True)
    surface.compute_normals(auto_orient_normals=True, inplace=True)
    data_names: list[str] = [
        name for name in source.point_data if name.startswith("expression")
    ]
    surface = melon.transfer_tri_point(
        source,
        surface,
        data=data_names,
        fill=0.0,
        nearest=melon.NearestPointOnSurface(normal_threshold=-0.5),
    )
    tetmesh = melon.transfer_tri_point_to_tet(
        surface, tetmesh, data=data_names, fill=0.0, point_id="point-id"
    )

    full: pv.PolyData = melon.load_polydata(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    cranium: pv.PolyData = melon.tri.extract_groups(full, groups["cranium"])
    mandible: pv.PolyData = melon.tri.extract_groups(full, groups["mandible"])
    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    cranium.cell_data["is-cranium"] = np.ones((cranium.n_cells,), dtype=np.bool)
    cranium.cell_data["is-face"] = np.zeros((cranium.n_cells,), dtype=np.bool)
    cranium.cell_data["is-mandible"] = np.zeros((cranium.n_cells,), dtype=np.bool)
    mandible.cell_data["is-cranium"] = np.zeros((mandible.n_cells,), dtype=np.bool)
    mandible.cell_data["is-face"] = np.zeros((mandible.n_cells,), dtype=np.bool)
    mandible.cell_data["is-mandible"] = np.ones((mandible.n_cells,), dtype=np.bool)
    skin.cell_data["is-cranium"] = np.zeros((skin.n_cells,), dtype=np.bool)
    skin.cell_data["is-face"] = np.ones((skin.n_cells,), dtype=np.bool)
    skin.cell_data["is-mandible"] = np.zeros((skin.n_cells,), dtype=np.bool)
    exclude_groups: list[str] = [
        "Ear",
        "EarNeckBack",
        "EarSocket EyeSocketTop",
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
    ]
    face_mask: Bool[np.ndarray, " face_cells"] = melon.tri.select_groups(
        skin, exclude_groups, invert=True
    )
    skin.cell_data["is-face"] &= face_mask
    source: pv.PolyData = pv.merge([cranium, mandible, skin])
    surface = melon.transfer_tri_cell_to_point_category(
        source,
        surface,
        data=["is-cranium", "is-mandible", "is-face"],
        fill=False,
        nearest=melon.NearestPointOnSurface(normal_threshold=-np.inf),
    )
    tetmesh = melon.transfer_tri_point_to_tet(
        surface,
        tetmesh,
        data=["is-cranium", "is-mandible", "is-face"],
        fill=False,
        point_id="point-id",
    )
    tetmesh.point_data["is-skull"] = (
        tetmesh.point_data["is-cranium"] | tetmesh.point_data["is-mandible"]
    )
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.run(main)
