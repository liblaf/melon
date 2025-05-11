from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    floting: Sequence[int | str] = [
        "Caruncle",
        "EarSocket EyeSocketTop",
        "EyeSocketBottom",
        "EyeSocketTop",
        "LipInnerBottom",
        "LipInnerTop",
        "MouthSocketBottom",
        "MouthSocketTop",
    ]
    output: Path = Path("data/02-intermediate/skin-with-mouth-socket.ply")
    source: Path = Path("data/01-raw/XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj")
    target: Path = Path("data/02-intermediate/skin.ply")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.source)
    source: pv.PolyData = melon.load_poly_data(cfg.source)
    cherries.log_input(cfg.target)
    target: pv.PolyData = melon.load_poly_data(cfg.target)
    cherries.log_input(melon.io.get_landmarks_path(cfg.source))
    source_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.source)
    cherries.log_input(melon.io.get_landmarks_path(cfg.target))
    target_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.target)

    free_polygons_floating: Integer[np.ndarray, " F"] = melon.triangle.select_groups(
        source, cfg.floting
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
    cherries.log_output(cfg.output)
    melon.save_landmarks(cfg.output, target_landmarks)
    cherries.log_output(melon.io.get_landmarks_path(cfg.output))


if __name__ == "__main__":
    cherries.run(main)
