from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    input: Path = Path("data/01-raw/full-human-head-anatomy.obj")
    output: Path = Path("data/02-intermediate/skin.ply")


@cherries.main()
def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.input)
    skin_left: pv.PolyData = melon.triangle.extract_cells(
        full, melon.triangle.select_groups(full, "skin_Cross_section")
    )
    skin_right: pv.PolyData = melon.extract_cells(
        full, melon.triangle.select_groups(full, "Skin001")
    )
    left_xmin: float = skin_left.bounds[0]
    right_xmax: float = skin_right.bounds[1]
    skin_left.translate([right_xmax - left_xmin, 0.0, 0.0], inplace=True)
    skin: pv.PolyData = pv.merge([skin_left, skin_right])
    edge_lengths: Float[np.ndarray, " E"] = melon.triangle.compute_edge_lengths(skin)
    skin.clean(tolerance=0.1 * edge_lengths.min(), absolute=False, inplace=True)
    melon.save(cfg.output, skin)


if __name__ == "__main__":
    cherries.run(main)
