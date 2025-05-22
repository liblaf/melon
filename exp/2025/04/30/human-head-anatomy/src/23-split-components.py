from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    face: Path = cherries.data("02-intermediate/skin-with-mouth-socket.ply")
    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.data("02-intermediate/groups.toml")
    tetgen: Path = cherries.data("02-intermediate/20-tetgen.vtu")

    output: Path = cherries.data("02-intermediate/23-tetgen.vtu")


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


def trimesh_point_data_to_tetmesh(
    tetmesh: pv.UnstructuredGrid, surface: pv.PolyData, data_name: str
) -> pv.UnstructuredGrid:
    point_ids: Integer[np.ndarray, " N"] = surface.point_data["point-id"]
    data: Bool[np.ndarray, " N"] = np.zeros((tetmesh.n_points,), dtype=bool)
    data[point_ids[surface.point_data[data_name]]] = True
    tetmesh.point_data[data_name] = data
    return tetmesh


def main(cfg: Config) -> None:
    face: pv.PolyData = melon.load_poly_data(cfg.face)
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)

    mesh.extract_largest(inplace=True)
    mesh.point_data["point-id"] = np.arange(mesh.n_points)

    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    mandible: pv.PolyData = melon.triangle.extract_groups(full, groups["mandible"])
    not_mandible: pv.PolyData = pv.merge(
        [
            face,
            melon.triangle.extract_groups(full, groups["cranium"]),
        ]
    )
    skull: pv.PolyData = melon.triangle.extract_groups(
        full, groups["cranium"] + groups["mandible"]
    )
    surface.point_data["is-face"] = classify(surface, face, skull)
    surface.point_data["is-mandible"] = classify(surface, mandible, not_mandible)
    surface.point_data["is-skull"] = classify(surface, skull, face)
    surface.point_data["is-cranium"] = (
        surface.point_data["is-skull"] & ~surface.point_data["is-mandible"]
    )
    assert not np.any(surface.point_data["is-face"] & surface.point_data["is-skull"])
    assert np.all(surface.point_data["is-face"] | surface.point_data["is-skull"])
    mesh = trimesh_point_data_to_tetmesh(mesh, surface, "is-cranium")
    mesh = trimesh_point_data_to_tetmesh(mesh, surface, "is-face")
    mesh = trimesh_point_data_to_tetmesh(mesh, surface, "is-mandible")
    mesh = trimesh_point_data_to_tetmesh(mesh, surface, "is-skull")
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
