from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    groups: Sequence[int | str] = ["asdf"]
    output: Path = Path("data/02_intermediate/skin-with-mouth-socket.ply")
    source: Path = Path("data/02-intermediate/human-head-anatomy.obj")
    target: Path = Path("data/02-intermediate/skin.ply")


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
    )
    melon.save(cfg.output, result)


if __name__ == "__main__":
    cherries.run(main)
