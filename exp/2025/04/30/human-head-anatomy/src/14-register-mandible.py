from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = cherries.input("00-sculptor-mandible.ply")
    target: Path = cherries.input("13-mandible.vtp")
    output: Path = cherries.output("14-mandible.vtp")


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_polydata(cfg.source)
    source.clean(inplace=True)
    target: pv.PolyData = melon.load_polydata(cfg.target)
    source_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.source)
    target_landmarks: Float[np.ndarray, "L 3"] = melon.load_landmarks(cfg.target)
    result: pv.PolyData = melon.tri.fast_wrapping(
        source,
        target,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
    )

    result = melon.transfer.transfer_tri_cell_to_point_category(
        target,
        result,
        data="GroupId",
        fill=-1,
        nearest=melon.proximity.NearestPointOnSurface(normal_threshold=None),
    )
    result.field_data["GroupName"] = target.field_data["GroupName"]

    melon.save(cfg.output, result)
    melon.save_landmarks(cfg.output, target_landmarks)


if __name__ == "__main__":
    cherries.main(main)
