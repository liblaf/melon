from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = cherries.input(
        "01-raw/sculptor/mandible.ply", extra=melon.io.get_landmarks_path
    )
    target: Path = cherries.input(
        "02-intermediate/13-mandible.vtp", extra=melon.io.get_landmarks_path
    )
    output: Path = cherries.output(
        "02-intermediate/14-mandible.ply", extra=melon.io.get_landmarks_path
    )


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
        verbose=True,
    )

    melon.save(cfg.output, result)
    melon.save_landmarks(cfg.output, target_landmarks)


if __name__ == "__main__":
    cherries.run(main)
