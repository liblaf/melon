from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


def classify(
    surface: pv.PolyData, include: pv.PolyData, exclude: pv.PolyData
) -> Bool[np.ndarray, " N"]:
    nearest_algo: melon.NearestAlgorithm = melon.NearestPointOnSurface(
        distance_threshold=np.inf,
        fallback_to_nearest_vertex=False,
        normal_threshold=-np.inf,
    )
    nearest_include: melon.NearestResult = melon.nearest(
        include, surface, algo=nearest_algo
    )
    nearest_exclude: melon.NearestResult = melon.nearest(
        exclude, surface, algo=nearest_algo
    )
    is_include: Bool[np.ndarray, " N"] = (
        nearest_include["distance"] <= nearest_exclude["distance"]
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
    skin: Path = cherries.input("02-intermediate/12-skin.vtp")
    cranium: Path = cherries.input("02-intermediate/14-cranium.ply")
    mandible: Path = cherries.input("02-intermediate/14-mandible.ply")

    tetgen: Path = cherries.input("02-intermediate/34-tetgen.vtu")
    source: Path = cherries.input("40-expression-flame.vtp")

    output: Path = cherries.output("40-expression.vtu")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(cfg.tetgen)
    source: pv.PolyData = melon.io.load_polydata(cfg.source)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]

    surface = surface.sample(
        source,
        tolerance=0.01 * surface.length,
        snap_to_closest_point=True,
    )  # pyright: ignore[reportAssignmentType]
    ic(surface.point_data["displacement"])
    tetmesh = melon.tetra.transfer_point_data_from_surface(
        surface,
        tetmesh,
        data=["displacement"],
        fill=0.0,
        # nearest=melon.proximity.NearestPoint(normal_threshold=0.0),
    )

    skin: pv.PolyData = melon.load_polydata(cfg.skin)
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
    face: pv.PolyData = melon.tri.extract_groups(skin, exclude_groups, invert=True)
    not_face: pv.PolyData = pv.merge(
        [
            melon.io.load_polydata(cfg.cranium),
            melon.io.load_polydata(cfg.mandible),
            melon.tri.extract_groups(skin, exclude_groups),
        ]
    )
    surface.point_data["is-face"] = classify(surface, face, not_face)

    tetmesh.point_data["is-skin"] = tetmesh.point_data["is-face"]
    tetmesh = transfer_trimesh_point_data_to_tetmesh(tetmesh, surface, "is-face")
    melon.io.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.run(main)
