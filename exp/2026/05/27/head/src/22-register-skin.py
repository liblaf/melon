import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    floating: Path = cherries.input("00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj")
    floating_gingiva_landmarks: Path = cherries.input(
        "00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.gingiva.landmarks.json"
    )
    floating_skin_landmarks: Path = cherries.input(
        "00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.skin.landmarks.json"
    )
    fixed_eye: Path = cherries.input("20-eye.ply")
    fixed_gingiva: Path = cherries.input("20-gingiva.ply")
    fixed_skin: Path = cherries.input("20-skin-smoothed.ply")
    output: Path = cherries.output("22-skin.vtp")


FREE_GROUPS: list[str] = [
    "Caruncle",
    "EarSocket EyeSocketTop",
    "EyeSocketBottom",
    "EyeSocketTop",
]
MOUTH_SOCKET_GROUPS: list[str] = [
    "MouthSocketBottom",
    "MouthSocketTop",
]


def filter_mouth_socket(
    floating: pv.PolyData,
    fixed: pv.PolyData,
    floating_landmarks: Float[np.ndarray, "L 3"],
    fixed_landmarks: Float[np.ndarray, "L 3"],
) -> Integer[np.ndarray, " N"]:
    floating.cell_data["CellId"] = np.arange(floating.n_cells)
    print(f"{floating.length=}")
    fixed_wrapped: pv.PolyData = melon.ext.fast_wrapping(
        floating=fixed,
        fixed=floating,
        floating_landmarks=fixed_landmarks,
        fixed_landmarks=floating_landmarks,
    )
    melon.save(fixed_wrapped, cherries.temp("22-gingiva-wrapped.vtp"))
    cell_centers: pv.PolyData = floating.cell_centers()
    cell_centers.compute_implicit_distance(fixed_wrapped, inplace=True)
    free_mask: Bool[np.ndarray, " N"] = (
        np.abs(cell_centers.point_data["implicit_distance"]) > 1e-2 * floating.length
    )
    floating.cell_data["implicit_distance"] = cell_centers.point_data[
        "implicit_distance"
    ]
    melon.save(floating, cherries.temp("22-floating-mouth-socket.vtp"))
    return floating.cell_data["CellId"][free_mask]


def main(cfg: Config) -> None:
    floating: pv.PolyData = melon.io.load_polydata(cfg.floating)
    floating.clean(inplace=True)
    floating.flip_faces(inplace=True)
    floating.triangulate(inplace=True)
    floating_gingiva_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(
        cfg.floating_gingiva_landmarks
    )
    floating_skin_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(
        cfg.floating_skin_landmarks
    )
    floating_landmarks: Float[np.ndarray, "L 3"] = np.concat(
        [floating_gingiva_landmarks, floating_skin_landmarks], axis=0
    )

    fixed_eye: pv.PolyData = melon.io.load_polydata(cfg.fixed_eye)
    fixed_eye.flip_faces(inplace=True)
    fixed_gingiva: pv.PolyData = melon.io.load_polydata(cfg.fixed_gingiva)
    fixed_gingiva.flip_faces(inplace=True)
    fixed_skin: pv.PolyData = melon.io.load_polydata(cfg.fixed_skin)
    fixed: pv.PolyData = pv.merge([fixed_eye, fixed_gingiva, fixed_skin])
    fixed_gingiva_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(
        cfg.fixed_gingiva
    )
    fixed_skin_landmarks: Float[np.ndarray, "L 3"] = melon.io.load_landmarks(
        cfg.fixed_skin
    )
    fixed_landmarks: Float[np.ndarray, "L 3"] = np.concat(
        [fixed_gingiva_landmarks, fixed_skin_landmarks], axis=0
    )

    free_polygons: Integer[np.ndarray, " N"] = melon.tri.select_groups(
        floating, FREE_GROUPS
    )
    mouth_socket_free_polygons: Integer[np.ndarray, " N"] = filter_mouth_socket(
        floating=melon.tri.extract_groups(floating, MOUTH_SOCKET_GROUPS),
        fixed=fixed_gingiva,
        floating_landmarks=floating_gingiva_landmarks,
        fixed_landmarks=fixed_gingiva_landmarks,
    )
    free_polygons: Integer[np.ndarray, " N"] = np.append(
        free_polygons, mouth_socket_free_polygons, axis=0
    )
    floating.cell_data["IsFree"] = np.isin(np.arange(floating.n_cells), free_polygons)

    result: pv.PolyData = melon.ext.fast_wrapping(
        floating,
        fixed,
        floating_landmarks=floating_landmarks,
        fixed_landmarks=fixed_landmarks,
        free_polygons_floating=free_polygons,
    )
    melon.save(result, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
