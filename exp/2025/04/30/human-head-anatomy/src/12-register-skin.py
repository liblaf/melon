from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries
from liblaf.melon import compat


class Config(cherries.BaseConfig):
    source: Path = cherries.input("00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj")
    target: Path = cherries.input("11-skin.ply")

    output: Path = cherries.output("12-skin.vtp")


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_polydata(cfg.source)
    source.clean(inplace=True)
    source.field_data["GroupName"] = [
        cast("str", name).split(maxsplit=1)[0] for name in compat.get_group_name(source)
    ]

    target: pv.PolyData = melon.load_polydata(cfg.target)
    source_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.source)
    target_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.target)

    free_polygons_floating: Integer[np.ndarray, " F"] = melon.tri.select_groups(
        source,
        [
            "Caruncle",
            "EarSocket",
            "EyeSocketBottom",
            "EyeSocketTop",
            # "LipInnerBottom",
            # "LipInnerTop",
            "MouthSocketBottom",
            "MouthSocketTop",
        ],
    )
    result: pv.PolyData = melon.tri.fast_wrapping(
        source,
        target,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        free_polygons_floating=free_polygons_floating,
    )
    melon.save(cfg.output, result)
    melon.save_landmarks(cfg.output, target_landmarks)


if __name__ == "__main__":
    cherries.main(main)
