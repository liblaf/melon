from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    groups: Sequence[int | str] = [
        "Caruncle",
        "EarSocket EyeSocketTop",
        "EyeSocketBottom",
        "EyeSocketTop",
        "LipInnerBottom",
        "LipInnerTop",
        "MouthSocketBottom",
        "MouthSocketTop",
    ]
    output: Path = Path("data/03-primary/skin-with-mouth-socket.ply")
    source: Path = Path("data/01-raw/XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj")
    target: Path = Path("data/03-primary/skin.ply")


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_poly_data(cfg.source)
    target: pv.PolyData = melon.load_poly_data(cfg.target)
    source_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.source)
    target_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.target)
    free_polygons_floating: Integer[np.ndarray, " F"] = melon.triangle.select_groups(
        source, cfg.groups
    )
    result: pv.PolyData = melon.fast_wrapping(
        source,
        target,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        free_polygons_floating=free_polygons_floating,
        verbose=True,
    )
    melon.save(cfg.output, result)
    melon.save_landmarks(cfg.output, target_landmarks)


if __name__ == "__main__":
    cherries.run(main)
