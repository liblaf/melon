# ruff: noqa: N806

from pathlib import Path

import numpy as np
import pyvista as pv
import scipy
import trimesh as tm
from jaxtyping import Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("14-cranium.vtp")
    mandible: Path = cherries.input("14-mandible.vtp")
    muscles: Path = cherries.input("20-muscles.vtm")
    skin: Path = cherries.input("11-skin.ply")


MUSCLE_NAMES: list[str] = [
    "Auricularis_anterior001_00",
    "Auricularis_anterior001_01",
    "Depressor_anguli001_00",
    "Depressor_anguli001_01",
    "Depressor_labii_inferioris001_00",
    "Depressor_labii_inferioris001_01",
    "Depressor_septi001",
    "Depressor_supercilli001_00",
    "Depressor_supercilli001_01",
    "Levator_anguli_oris001_00",
    "Levator_anguli_oris001_01",
    "Levator_labii_superioris_alaeque_nasi001_00",
    "Levator_labii_superioris_alaeque_nasi001_01",
    "Levator_labii_superioris001_00",
    "Levator_labii_superioris001_01",
    "Mentalis001_00",
    "Mentalis001_01",
    "Nasalis_alarportion001_00",
    "Nasalis_alarportion001_01",
    "Nasalis_transverse_portion001",
    "Occipitofrontalis_epicranius001",
    "Orbicularis_oculi001_00",
    "Orbicularis_oculi001_01",
    "Platysma001_00",
    "Platysma001_01",
    "Procerus001",
    "Risorius001_00",
    "Risorius001_01",
    "Temporal_fascia001_00",
    "Temporal_fascia001_01",
    "Zygomaticus_major001_00",
    "Zygomaticus_major001_01",
    "Zygomaticus_minor001_00",
    "Zygomaticus_minor001_01",
    # unsure
    "Masseter_deep001_00",
    "Masseter_deep001_01",
    "Masseter_superficial001_00",
    "Masseter_superficial001_01",
    "Orbicularis_oris001",
]

THICKNESS_THRESHOLD: float = 2.2  # centimeters


def select_muscles(muscles: pv.MultiBlock) -> pv.MultiBlock:
    selected: pv.MultiBlock = pv.MultiBlock([muscles[name] for name in MUSCLE_NAMES])
    return selected


def skin_to_skull(
    skin: pv.PolyData, skull: pv.PolyData
) -> Float[np.ndarray, " points"]:
    skull_tm: tm.Trimesh = melon.as_trimesh(skull)
    locations: Float[np.ndarray, "M 3"]
    index_ray: Integer[np.ndarray, " M"]
    locations, index_ray, _index_tri = skull_tm.ray.intersects_location(
        ray_origins=skin.points, ray_directions=-skin.point_normals
    )
    thickness: Float[np.ndarray, " points"] = np.full(skin.n_points, np.inf)
    distance: Float[np.ndarray, " M"] = np.linalg.norm(
        locations - skin.points[index_ray], axis=-1
    )
    np.minimum.at(thickness, index_ray, distance)
    thickness[~np.isfinite(thickness)] = np.nan
    thickness[thickness > THICKNESS_THRESHOLD] = np.nan
    return thickness


def skin_to_muscle(
    skin: pv.PolyData, muscle: pv.PolyData
) -> tuple[Float[np.ndarray, " points"], Float[np.ndarray, " points"]]:
    skin_to_skull: Float[np.ndarray, " points"] = skin.point_data["SkinToSkull"]
    skin_to_skull = np.nan_to_num(skin_to_skull, nan=np.inf)
    muscle_tm: tm.Trimesh = melon.as_trimesh(muscle)
    locations: Float[np.ndarray, "M 3"]
    index_ray: Integer[np.ndarray, " M"]
    locations, index_ray, _index_tri = muscle_tm.ray.intersects_location(
        ray_origins=skin.points, ray_directions=-skin.point_normals
    )
    ray_distance: Float[np.ndarray, " M"] = np.linalg.norm(
        locations - skin.points[index_ray], axis=-1
    )
    thickness_min: Float[np.ndarray, " points"] = np.full((skin.n_points,), np.inf)
    thickness_max: Float[np.ndarray, " points"] = np.zeros((skin.n_points,))
    keep: Bool[np.ndarray, " M"] = (ray_distance < skin_to_skull[index_ray] + 0.5) & (
        ray_distance < THICKNESS_THRESHOLD
    )
    idx: Integer[np.ndarray, " K"] = index_ray[keep]
    dist: Float[np.ndarray, " K"] = ray_distance[keep]
    np.minimum.at(thickness_min, idx, dist)
    np.maximum.at(thickness_max, idx, dist)
    thickness_max[thickness_max < thickness_min] = np.nan
    thickness_min[~np.isfinite(thickness_min)] = np.nan

    # thickness_min[thickness_min > THICKNESS_THRESHOLD] = np.nan
    # thickness_max[thickness_max > THICKNESS_THRESHOLD] = np.nan

    return thickness_min, thickness_max


def smooth(skin: pv.PolyData, lambda_: float = 1.0) -> pv.PolyData:
    thickness_min: Float[np.ndarray, " points"] = skin.point_data["SkinToMuscleMin"]
    thickness_max: Float[np.ndarray, " points"] = skin.point_data["SkinToMuscleMax"]
    L: scipy.sparse.coo_matrix = tm.smoothing.laplacian_calculation(
        melon.as_trimesh(skin)
    )  # pyright: ignore[reportAssignmentType]
    mask: Bool[np.ndarray, " points"] = np.isfinite(thickness_min)
    valid_idx: Integer[np.ndarray, " K"] = np.flatnonzero(mask)
    W: scipy.sparse.coo_matrix = scipy.sparse.coo_matrix(
        (
            np.full(valid_idx.shape, np.sqrt(lambda_)),
            (np.arange(valid_idx.size), valid_idx),
        ),
        (valid_idx.size, skin.n_points),
    )
    A: Float[scipy.sparse.coo_matrix, "M N"] = scipy.sparse.vstack(
        [L - scipy.sparse.identity(L.shape[0]), W]
    )
    b: Float[np.ndarray, " M"] = np.concat(
        [np.zeros((skin.n_points,)), thickness_min[mask]], axis=0
    )
    bounds: scipy.optimize.Bounds = scipy.optimize.Bounds(
        np.zeros((skin.n_points,)), np.nan_to_num(thickness_min, nan=np.inf)
    )
    result: scipy.optimize.OptimizeResult = scipy.optimize.lsq_linear(
        A,
        b,
        bounds=bounds,  # pyright: ignore[reportArgumentType]
        verbose=2,
    )
    ic(result)
    skin.point_data["SkinToMuscleMinSmooth"] = result["x"]

    mask: Bool[np.ndarray, " points"] = np.isfinite(thickness_max)
    valid_idx: Integer[np.ndarray, " K"] = np.flatnonzero(mask)
    W: scipy.sparse.coo_matrix = scipy.sparse.coo_matrix(
        (
            np.full(valid_idx.shape, np.sqrt(lambda_)),
            (np.arange(valid_idx.size), valid_idx),
        ),
        (valid_idx.size, skin.n_points),
    )
    A: Float[scipy.sparse.coo_matrix, "M N"] = scipy.sparse.vstack(
        [L - scipy.sparse.identity(L.shape[0]), W]
    )
    b: Float[np.ndarray, " M"] = np.concat(
        [np.zeros((skin.n_points,)), thickness_max[mask]], axis=0
    )
    bounds: scipy.optimize.Bounds = scipy.optimize.Bounds(
        np.nan_to_num(thickness_max, nan=0.0), np.full((skin.n_points,), np.inf)
    )
    result = scipy.optimize.lsq_linear(
        A,
        b,
        bounds=bounds,  # pyright: ignore[reportArgumentType]
        verbose=2,
    )
    ic(result)
    skin.point_data["SkinToMuscleMaxSmooth"] = result["x"]

    return skin


def smooth_iter(
    skin: pv.PolyData,
    lamb: float = 0.5,
    n_iter_fill: int = 1000,
    n_iter_smooth: int = 1000,
) -> pv.PolyData:
    skin_to_skull: Float[np.ndarray, " points"] = skin.point_data["SkinToSkull"]
    hit_skull: Bool[np.ndarray, " points"] = np.isfinite(skin_to_skull)

    L: scipy.sparse.coo_matrix = tm.smoothing.laplacian_calculation(
        melon.as_trimesh(skin)
    )  # pyright: ignore[reportAssignmentType]
    thickness_min: Float[np.ndarray, " points"] = skin.point_data["SkinToMuscleMin"]
    nan_mask: Bool[np.ndarray, " points"] = ~np.isfinite(thickness_min)
    thickness_min_max: float = np.nanmax(thickness_min)
    thickness_min = np.nan_to_num(thickness_min, nan=thickness_min_max)
    thickness_min[hit_skull] = np.minimum(
        thickness_min[hit_skull], skin_to_skull[hit_skull]
    )
    thickness_min_init: Float[np.ndarray, " points"] = thickness_min.copy()
    for _ in range(n_iter_fill):
        delta = L @ thickness_min - thickness_min
        thickness_min[nan_mask] += lamb * delta[nan_mask]
    for _ in range(n_iter_smooth):
        delta = L @ thickness_min - thickness_min
        thickness_min += 0.5 * delta
        thickness_min = np.minimum(thickness_min, thickness_min_init)
    skin.point_data["SkinToMuscleMinSmooth"] = thickness_min

    thickness_max: Float[np.ndarray, " points"] = skin.point_data["SkinToMuscleMax"]
    nan_mask: Bool[np.ndarray, " points"] = ~np.isfinite(thickness_max)
    thickness_max_min: float = np.nanmin(thickness_max)
    thickness_max = np.nan_to_num(thickness_max, nan=thickness_max_min)
    thickness_max_init: Float[np.ndarray, " points"] = thickness_max.copy()
    for _ in range(n_iter_fill):
        delta = L @ thickness_max - thickness_max
        thickness_max[nan_mask] += lamb * delta[nan_mask]
    for _ in range(n_iter_smooth):
        delta = L @ thickness_max - thickness_max
        thickness_max += 0.5 * delta
        thickness_max = np.maximum(thickness_max, thickness_max_init)
        thickness_max[hit_skull] = np.minimum(
            thickness_max[hit_skull], skin_to_skull[hit_skull]
        )
    skin.point_data["SkinToMuscleMaxSmooth"] = thickness_max
    return skin


def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)
    muscles: pv.MultiBlock = pv.read(cfg.muscles)  # pyright: ignore[reportAssignmentType]
    muscles = select_muscles(muscles)
    muscles.save(cherries.temp("21-muscles-selected.vtm"))

    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    skin.triangulate(inplace=True)
    skin.subdivide_adaptive(
        max_edge_len=0.1,  # centimeters
        inplace=True,
    )

    skin.point_data["SkinToSkull"] = skin_to_skull(skin, pv.merge([cranium, mandible]))
    skin.point_data["SkinToMuscleMin"], skin.point_data["SkinToMuscleMax"] = (
        skin_to_muscle(skin, muscles.combine().extract_surface())
    )
    melon.save(cherries.temp("21-skin-thickness.vtp"), skin)
    # skin = smooth(skin, lambda_=1e-2)
    skin = smooth_iter(skin)
    melon.save(cherries.output("21-skin-thickness.vtp"), skin)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
