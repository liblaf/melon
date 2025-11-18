from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    input: Path = cherries.input("00-Full human head anatomy.obj")
    output: Path = cherries.output("11-skin.ply")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.input)
    full.clean(inplace=True)
    skin_left: pv.PolyData = melon.tri.extract_groups(full, "Skin001")
    skin_right: pv.PolyData = melon.tri.extract_groups(full, "skin_Cross_section")
    left_xmax: float = skin_left.bounds[1]
    right_xmin: float = skin_right.bounds[0]
    skin_right.translate([left_xmax - right_xmin, 0.0, 0.0], inplace=True)
    skin: pv.PolyData = pv.merge([skin_right, skin_left])
    edge_lengths: Float[np.ndarray, " E"] = melon.tri.compute_edge_lengths(skin)
    skin.clean(tolerance=0.5 * edge_lengths.min(), inplace=True, absolute=True)
    melon.save(cfg.output, skin)


if __name__ == "__main__":
    cherries.main(main)
