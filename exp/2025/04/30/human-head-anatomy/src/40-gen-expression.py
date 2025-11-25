from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = cherries.input("40-expression-flame.vtp")
    tetmesh: Path = cherries.input("31-masks.vtu")

    output: Path = cherries.output("40-expression.vtu")


def geodestic_transform(mesh: pv.PolyData) -> pv.PolyData:
    archors: Float[np.ndarray, "a 3"] = np.asarray(
        [
            [0.72436285, 27.4310532, 9.97920895],
            [0.724529326, 25.2615795, 9.52740097],
            [-1.72519755, 26.2093906, 8.44369125],
        ]
    )
    nearest_result: melon.NearestPointResult = melon.nearest(mesh, archors)
    archor_idx: Integer[np.ndarray, " a"] = nearest_result.vertex_id
    geo_dist: Float[np.ndarray, "p a"] = melon.geodesic_distance(mesh, archor_idx)
    geo_mesh: pv.PolyData = pv.PolyData.from_regular_faces(geo_dist, mesh.regular_faces)
    geo_mesh.copy_attributes(mesh)
    return geo_mesh


def transfer_by_geodestic(
    source: pv.PolyData, target: pv.PolyData, data_names: list[str]
) -> pv.PolyData:
    source_geo: pv.PolyData = geodestic_transform(source)
    target_geo: pv.PolyData = geodestic_transform(target)
    target_geo = melon.transfer_tri_point(
        source_geo,
        target_geo,
        data=data_names,
        fill=np.nan,
        nearest=melon.NearestPointOnSurface(
            distance_threshold=0.01, normal_threshold=None
        ),
    )
    target = target.copy()
    target.copy_attributes(target_geo)
    return target


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_polydata(cfg.source)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    tetmesh.point_data["_PointId"] = np.arange(tetmesh.n_points)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    source = melon.fast_wrapping(source, surface)
    source.compute_normals(auto_orient_normals=True, inplace=True)
    surface.compute_normals(auto_orient_normals=True, inplace=True)
    data_names: list[str] = [
        name for name in source.point_data if name.startswith("Expression")
    ]
    surface = melon.transfer_tri_point(
        source,
        surface,
        data=data_names,
        fill=np.nan,
        nearest=melon.NearestPointOnSurface(
            distance_threshold=0.01, normal_threshold=0.8
        ),
    )
    melon.save(cherries.temp("40-expression-transfer-nearest.vtp"), surface)
    surface_geo: pv.PolyData = transfer_by_geodestic(source, surface, data_names)
    melon.save(cherries.temp("40-expression-transfer-geo.vtp"), surface_geo)
    for name in data_names:
        data: np.ndarray = surface.point_data[name]
        missing: Bool[np.ndarray, " ..."] = np.isnan(data)
        surface.point_data[name] = np.nan_to_num(
            np.where(missing, surface_geo.point_data[name], data), nan=0.0
        )

    tetmesh = melon.transfer_tri_point_to_tet(
        surface, tetmesh, data=data_names, fill=0.0, point_id="_PointId"
    )
    del tetmesh.point_data["_PointId"]
    skull_mask = tetmesh.point_data["IsCranium"] | tetmesh.point_data["IsMandible"]
    for name in data_names:
        tetmesh.point_data[name][skull_mask] = 0.0
    melon.save(cfg.output, tetmesh)

    face: pv.PolyData = surface.extract_points(
        surface.point_data["IsFace"]
    ).extract_surface()
    melon.save(cherries.temp("40-expression-face.vtp"), face)


if __name__ == "__main__":
    cherries.main(main)
