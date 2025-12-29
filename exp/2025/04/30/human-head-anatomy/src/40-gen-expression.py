from pathlib import Path

import numpy as np
import potpourri3d as pp3d
import pyvista as pv
import scipy
import trimesh as tm
from jaxtyping import Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries

SUFFIX: str = "-515k"


class Config(cherries.BaseConfig):
    source: Path = cherries.input("40-expression-flame.vtp")
    tetmesh: Path = cherries.input(f"31-masks{SUFFIX}.vtu")

    output: Path = cherries.output(f"40-expression{SUFFIX}.vtu")


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


def inpaint_by_heat(
    mesh: pv.PolyData, data_names: list[str], mask: Bool[np.ndarray, " points"]
) -> pv.PolyData:
    ic(mesh.is_manifold)
    mesh = melon.tri.extract_points(mesh, mask, adjacent_cells=True)
    ic(mesh.is_manifold)
    # mesh = melon.tri.fix_inversion(mesh)
    ic(np.count_nonzero(mesh.regular_faces[:, 0] == mesh.regular_faces[:, 1]))
    ic(np.count_nonzero(mesh.regular_faces[:, 0] == mesh.regular_faces[:, 2]))
    ic(np.count_nonzero(mesh.regular_faces[:, 1] == mesh.regular_faces[:, 2]))
    solver = pp3d.MeshVectorHeatSolver(mesh.points, mesh.regular_faces)
    for name in data_names:
        data: Float[np.ndarray, "points dim"] = mesh.point_data[name]
        for dim in range(data.shape[1]):
            data[:, dim] = solver.extend_scalar(~np.isnan(data[:, dim]), data[:, dim])  # pyright: ignore[reportArgumentType]
        mesh.point_data[name] = data
    return mesh


def inpaint_by_smoothing(mesh: pv.PolyData, data_names: list[str]) -> pv.PolyData:
    mesh = mesh.copy()
    mesh_tm: tm.Trimesh = melon.as_trimesh(mesh)
    for name in data_names:
        pinned_vertices: Integer[np.ndarray, " p"] = np.flatnonzero(
            np.all(np.isfinite(mesh.point_data[name]), axis=-1)
        )
        laplacian: scipy.sparse.coo_matrix = tm.smoothing.laplacian_calculation(
            mesh_tm, pinned_vertices=pinned_vertices
        )  # pyright: ignore[reportAssignmentType]
        deformed: tm.Trimesh = tm.Trimesh(
            mesh_tm.vertices + np.nan_to_num(mesh.point_data[name], nan=0.0),
            mesh_tm.faces,
        )
        smoothed: tm.Trimesh = tm.smoothing.filter_taubin(
            deformed, iterations=10**5, laplacian_operator=laplacian
        )
        mesh.point_data[name] = smoothed.vertices - mesh.points
    return mesh


def transfer_by_geodestic(
    source: pv.PolyData, target: pv.PolyData, data_names: list[str]
) -> pv.PolyData:
    max_tri_area: float = 5e-5 * source.length**2
    ic(source)
    source = source.subdivide_adaptive(max_tri_area=max_tri_area)  # pyright: ignore[reportAssignmentType]
    ic(source)
    target_dense: pv.PolyData = target.subdivide_adaptive(max_tri_area=max_tri_area)  # pyright: ignore[reportAssignmentType]
    source_geo: pv.PolyData = geodestic_transform(source)
    target_geo: pv.PolyData = geodestic_transform(target_dense)
    target_geo = melon.transfer_tri_point(
        source_geo,
        target_geo,
        data=data_names,
        fill=np.nan,
        nearest=melon.NearestPointOnSurface(
            distance_threshold=0.01, normal_threshold=None
        ),
    )
    original_point_id: Integer[np.ndarray, " p"] = melon.nearest(
        target_dense, target
    ).vertex_id
    for name in data_names:
        target.point_data[name] = target_geo.point_data[name][original_point_id]
    return target


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_polydata(cfg.source)
    source.extract_largest(inplace=True)  # remove eyeballs
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    tetmesh.point_data["_PointId"] = np.arange(tetmesh.n_points)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    source = melon.fast_wrapping(source, surface)
    melon.save(cherries.temp("40-expression-wrapped.vtp"), source)
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
    nearest: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        source, surface, distance_threshold=0.01, normal_threshold=None
    )
    inpaint_mask: Bool[np.ndarray, " p"] = np.any(
        ~nearest.missing[:, np.newaxis], axis=-1
    )
    surface_inpaint: pv.PolyData = inpaint_by_smoothing(surface, data_names)
    melon.save(cherries.temp("40-expression-transfer-inpaint.vtp"), surface_inpaint)
    # surface_geo: pv.PolyData = transfer_by_geodestic(source, surface, data_names)
    # melon.save(cherries.temp("40-expression-transfer-geo.vtp"), surface_geo)
    for name in data_names:
        data: Float[np.ndarray, "points dim"] = surface.point_data[name]
        missing: Bool[np.ndarray, " points"] = np.any(np.isnan(data), axis=-1)
        surface.point_data[name] = np.nan_to_num(
            np.where(
                np.expand_dims(missing & inpaint_mask, axis=-1),
                surface_inpaint.point_data[name],
                data,
            ),
            nan=0.0,
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
