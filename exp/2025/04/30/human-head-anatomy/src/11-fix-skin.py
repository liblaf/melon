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
    skin_left: pv.PolyData = melon.tri.extract_groups(full, "skin_Cross_section")
    skin_right: pv.PolyData = melon.tri.extract_groups(full, "Skin001")
    skin_left.translate(
        [skin_right.bounds.x_max - skin_left.bounds.x_min, 0.0, 0.0], inplace=True
    )
    skin: pv.PolyData = pv.merge([skin_left, skin_right])
    edge_lengths: Float[np.ndarray, " E"] = melon.compute_edges_length(skin)
    skin.clean(tolerance=0.5 * edge_lengths.min(), inplace=True, absolute=True)
    melon.save(cfg.output, skin)


if __name__ == "__main__":
    cherries.main(main)
