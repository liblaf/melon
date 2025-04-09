from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    face: Path = (
        grapes.find_project_dir() / "data/03-primary/skin-with-mouth-socket.ply"
    )
    full: Path = grapes.find_project_dir() / "data/01-raw/Full human head anatomy.obj"
    groups: Path = grapes.find_project_dir() / "data/02-intermediate/groups.toml"
    input: Path = grapes.find_project_dir() / "data/03-primary/tetgen.vtu"
    inputs_dir: Path = grapes.find_project_dir() / "data/03-primary/inputs"
    output: Path = grapes.find_project_dir() / "data/03-primary/tetgen-clean.vtu"
    outputs_dir: Path = grapes.find_project_dir() / "data/03-primary/tetgen"


def classify(
    surface: pv.PolyData, include: pv.PolyData, exclude: pv.PolyData
) -> Bool[np.ndarray, " N"]:
    nearest_algo: melon.NearestPointOnSurface = melon.NearestPointOnSurface(
        fallback_to_nearest_vertex=False, normal_threshold=-np.inf
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


def transfer_point_data(
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
    # groups["rigid-with-mandible"] = ["Mandibula_skull001"]
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)

    mesh.extract_largest(inplace=True)
    mesh.point_data["point-id"] = np.arange(mesh.n_points)

    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    mandible: pv.PolyData = melon.triangle.extract_groups(
        full, groups["rigid-with-mandible"]
    )
    not_mandible: pv.PolyData = pv.merge(
        [
            face,
            melon.triangle.extract_groups(
                full, set(groups["Skeletons"]) - set(groups["rigid-with-mandible"])
            ),
        ]
    )
    skull: pv.PolyData = melon.triangle.extract_groups(full, groups["Skeletons"])
    melon.save(cfg.inputs_dir / "face.ply", face)
    melon.save(cfg.inputs_dir / "mandible.ply", mandible)
    melon.save(cfg.inputs_dir / "skull.ply", skull)
    surface.point_data["is-face"] = classify(surface, face, skull)
    surface.point_data["is-mandible"] = classify(surface, mandible, not_mandible)
    surface.point_data["is-skull"] = classify(surface, skull, face)
    assert not np.any(surface.point_data["is-face"] & surface.point_data["is-skull"])
    assert np.all(surface.point_data["is-face"] | surface.point_data["is-skull"])
    mesh = transfer_point_data(mesh, surface, "is-face")
    mesh = transfer_point_data(mesh, surface, "is-mandible")
    mesh = transfer_point_data(mesh, surface, "is-skull")
    melon.save(cfg.output, mesh)

    for name in ["face", "mandible", "skull"]:
        component: pv.PolyData = melon.triangle.extract_points(
            surface, surface.point_data[f"is-{name}"]
        )
        melon.save(cfg.outputs_dir / f"{name}.vtp", component)


if __name__ == "__main__":
    cherries.run(main)
