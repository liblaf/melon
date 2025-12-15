from pathlib import Path

import numpy as np
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    skin: Path = cherries.input("12-skin.vtp")
    output: Path = cherries.temp("22-geodesic-path-skin.vtp")


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.load_polydata(cfg.skin)

    nearest_result: melon.NearestPointResult = melon.nearest(
        skin,
        np.asarray(
            [
                [0.724907, 25.935352, 9.666245],  # midpoint below lips
                [0.689024, 26.452383, 9.960980],  # midpoint above lips
            ]
        ),
    )
    source: int
    target: int
    source, target = nearest_result.vertex_id
    output: pv.PolyData = melon.geodesic_path(skin, source, target)

    melon.save(cfg.output, output)


if __name__ == "__main__":
    cherries.main(main, profile="playground")
