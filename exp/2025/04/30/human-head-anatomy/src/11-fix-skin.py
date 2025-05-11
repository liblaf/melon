from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    input: Path = cherries.data("01-raw/Full human head anatomy.obj")
    output: Path = cherries.data("02-intermediate/skin.ply")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.input)
    full: pv.PolyData = melon.load_poly_data(cfg.input)
    skin_left: pv.PolyData = melon.triangle.extract_groups(full, "Skin001")
    skin_right: pv.PolyData = melon.triangle.extract_groups(full, "skin_Cross_section")
    left_xmax: float = skin_left.bounds[1]
    right_xmin: float = skin_right.bounds[0]
    skin_right.translate([left_xmax - right_xmin, 0.0, 0.0], inplace=True)
    skin: pv.PolyData = pv.merge([skin_right, skin_left])
    edge_lengths: Float[np.ndarray, " E"] = melon.triangle.compute_edge_lengths(skin)
    skin.clean(tolerance=0.5 * edge_lengths.min(), inplace=True, absolute=True)
    melon.save(cfg.output, skin)
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.run(main)
