import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float, Integer

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    basemesh: Path = cherries.input("60-blendshapes.vtp")
    target: Path = cherries.input("22-skin.vtp")
    output: Path = cherries.output("61-blendshapes.vtp")


BASEMESH_FLOATING_GROUPS: list[str] = [
    "MouthSocketUpper",
    "MouthSocketLower",
    "Nostrils",
]
TARGET_EXCLUDE_GROUPS: list[str] = [
    "Caruncle",
    "EarSocket EyeSocketTop",
    "EyeSocketBottom",
    "EyeSocketTop",
    "LipInnerBottom",
    "LipInnerTop",
    "MouthSocketBottom",
    "MouthSocketTop",
]


def main(cfg: Config) -> None:
    basemesh: pv.PolyData = pv.read(cfg.basemesh, cls=pv.PolyData)
    basemesh_landmarks: Float[np.ndarray, "l 3"] = melon.io.load_landmarks(cfg.basemesh)
    target: pv.PolyData = pv.read(cfg.target, cls=pv.PolyData)
    target_landmarks: Float[np.ndarray, "l 3"] = melon.io.load_landmarks(cfg.target)
    target: pv.PolyData = melon.tri.extract_groups(
        target, TARGET_EXCLUDE_GROUPS, invert=True
    )
    free_polygons_floating: Bool[np.ndarray, " c"] = melon.tri.select_groups(
        basemesh, BASEMESH_FLOATING_GROUPS
    )
    free_polygons_floating: Integer[np.ndarray, " F"] = np.flatnonzero(
        free_polygons_floating
    )
    result: pv.PolyData = melon.ext.fast_wrapping(
        basemesh,
        target,
        floating_landmarks=basemesh_landmarks,
        fixed_landmarks=target_landmarks,
        free_polygons_floating=free_polygons_floating,
    )

    matrix, _transformed, cost = tm.registration.procrustes(
        target_landmarks, basemesh_landmarks
    )
    logger.info("procrustes cost: %f", cost)
    matrix_inv: Float[np.ndarray, "4 4"] = tm.transformations.inverse_matrix(matrix)
    floating_neutral: pv.PolyData = result.transform(matrix, inplace=False)
    for expression_name in basemesh.field_data["ExpressionName"]:
        ref_expression: pv.PolyData = basemesh.warp_by_vector(
            expression_name, inplace=False
        )
        delta: pv.PolyData = melon.ext.delta_transfer(
            floating_neutral, basemesh, ref_expression
        )
        delta.transform(matrix_inv, inplace=True)
        result.point_data[expression_name] = delta.points - result.points
    melon.save(result, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
