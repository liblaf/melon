from pathlib import Path

import einops
import numpy as np
import pyvista as pv
import scipy
import trimesh as tm
from jaxtyping import Bool, Float, Integer

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


def transfer_point_data_with_normal_threshold(
    source: pv.PolyData,
    target: pv.PolyData,
    data_name: str,
    normal_threshold: float = 0.5,
) -> pv.PolyData:
    tree = scipy.spatial.KDTree(source.points)
    neighbors = tree.query_ball_point(
        target.points, r=source.length * 1e-1, return_sorted=True, workers=-1
    )
    target.point_data[data_name] = np.zeros_like(target.points)
    for i in range(target.n_points):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            ic(i, "no neighbors found")
            continue
        target_normal = target.point_data["Normals"][i]
        source_normals = source.point_data["Normals"][nbrs]
        normal_dots = np.einsum("j,ij->i", target_normal, source_normals)
        valid_mask = normal_dots >= normal_threshold
        valid_nbrs = np.asarray(nbrs)[valid_mask]
        if len(valid_nbrs) == 0:
            ic(i, "no valid neighbors found")
            continue
        if len(valid_nbrs) < 3:
            ic(i, "too few valid neighbors found", len(valid_nbrs))
            continue
        barycentric = tm.triangles.points_to_barycentric(
            triangles=source.points[valid_nbrs[:3]][None],
            points=target.points[i][None],
        )
        transferred = einops.einsum(
            barycentric,
            source.point_data[data_name][valid_nbrs[:3]][None],
            "P B, P B ... -> P ...",
        )
        target.point_data[data_name][i] = transferred[0]
    return target


def transfer_point_data_with_normal_threshold_v2(
    source: pv.PolyData,
    target: pv.PolyData,
    data_name: str,
    normal_threshold: float = 0.5,
) -> pv.PolyData:
    target.point_data[data_name] = np.zeros_like(target.points)
    for i in range(target.n_points):
        target_normal = target.point_data["Normals"][i]
        source_face_mask = (
            np.vecdot(target_normal, source.face_normals) >= normal_threshold
        )
        source_valid = source.extract_cells(source_face_mask).extract_surface()
        source_valid_tm = melon.as_trimesh(source_valid)
        closest, _, triangle_id = source_valid_tm.nearest.on_surface(
            target.points[i][None]
        )
        barycentric = tm.triangles.points_to_barycentric(
            triangles=source_valid_tm.triangles[triangle_id],
            points=closest,
        )
        transferred = einops.einsum(
            barycentric,
            source_valid.point_data[data_name][source_valid_tm.faces[triangle_id]],
            "P B, P B ... -> P ...",
        )
        ic(target.point_data["__point-id"][i], transferred[0])
        target.point_data[data_name][i] = transferred[0]
    return target


def transfer_point_data(
    source: pv.PolyData, target: pv.PolyData, data_name: str
) -> pv.PolyData:
    source = source.connectivity("largest")
    source_raw = source
    source.compute_normals(inplace=True, auto_orient_normals=True)
    target.compute_normals(inplace=True, auto_orient_normals=True)
    source_raw.point_data["point-id"] = np.arange(source.n_points)
    melon.save("source.vtp", source_raw)
    data: np.ndarray = source.point_data[data_name]
    # points = tm.registration.nricp_amberg(
    #     melon.as_trimesh(source_raw), melon.as_trimesh(target)
    # )
    melon.save("source.obj", source)
    del source.point_data["Normals"]
    source.save("source.obj", recompute_normals=False)
    source: pv.PolyData = melon.fast_wrapping(source, target)
    source.copy_attributes(source_raw)
    # source.points = points
    source.compute_normals(inplace=True, auto_orient_normals=True)
    melon.save("wrapped.vtp", source)
    source_tm: tm.Trimesh = melon.io.as_trimesh(source)
    closest: Float[np.ndarray, "P 3"]
    distance: Float[np.ndarray, " P"]
    triangle_id: Integer[np.ndarray, " P"]
    closest, distance, triangle_id = source_tm.nearest.on_surface(target.points)
    barycentric: Float[np.ndarray, "P 3"] = tm.triangles.points_to_barycentric(
        triangles=source_tm.triangles[triangle_id], points=closest
    )
    transferred: Float[np.ndarray, "P ..."] = einops.einsum(
        barycentric, data[source_tm.faces[triangle_id]], "P B, P B ... -> P ..."
    )
    target.point_data[data_name] = transferred
    source_normal = source.face_normals[triangle_id]
    target_normal = target.point_normals
    normal_dot = np.einsum("ij,ij->i", source_normal, target_normal)
    normal_threshold = -0.5
    not_found_mask = (distance > 1e-2 * source_tm.scale) | (
        normal_dot < normal_threshold
    )
    not_found_mask &= (
        target.point_data["is-lip-bottom"] | target.point_data["is-lip-top"]
    )
    target.point_data[data_name][not_found_mask] = 0.0

    remaining_mask = (
        (distance < 1e-2 * source_tm.scale)
        & (normal_dot < normal_threshold)
        & (target.point_data["is-lip-bottom"] | target.point_data["is-lip-top"])
    )
    target.point_data["remaining-mask"] = remaining_mask
    if np.any(not_found_mask):
        target.point_data["__point-id"] = np.arange(target.n_points)
        remaining: pv.PolyData = pv.PointSet(target.points[remaining_mask])
        remaining.point_data["__point-id"] = target.point_data["__point-id"][
            remaining_mask
        ]
        remaining.point_data["Normals"] = target.point_normals[remaining_mask]
        remaining = transfer_point_data_with_normal_threshold_v2(
            source, remaining, data_name=data_name, normal_threshold=normal_threshold
        )
        target.point_data[data_name][remaining.point_data["__point-id"]] = (
            remaining.point_data[data_name]
        )
    target.point_data["not-found"] = not_found_mask
    return target


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(cfg.tetgen)
    source: pv.PolyData = melon.io.load_polydata(cfg.source)

    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]

    # surface = surface.sample(
    #     source,
    #     tolerance=0.01 * surface.length,
    #     snap_to_closest_point=True,
    # )  # pyright: ignore[reportAssignmentType]
    surface = transfer_point_data(source, surface, "displacement")
    melon.save("surface.vtp", surface)
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
