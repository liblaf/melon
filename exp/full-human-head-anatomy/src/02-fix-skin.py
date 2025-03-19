from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    groups: Path = Path("data/01_raw/groups.toml")
    input: Path = Path("data/01_raw/human-head-anatomy.obj")
    output: Path = Path("data/02_intermediate/")


@cherries.main()
def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.input)
    skin_left: pv.PolyData = melon.extract_cells(
        full, melon.select_cells_by_group(full, "skin_Cross_section")
    )
    skin_right: pv.PolyData = melon.extract_cells(
        full, melon.select_cells_by_group(full, "Skin001")
    )
    left_xmin: float = skin_left.bounds[0]
    right_xmax: float = skin_right.bounds[1]
    skin_left.translate([right_xmax - left_xmin, 0.0, 0.0], inplace=True)
    skin: pv.PolyData = pv.merge([skin_left, skin_right])
    skin.clean(tolerance=1e-3, absolute=False, inplace=True)
    melon.save(cfg.output / "skin.ply", skin)


if __name__ == "__main__":
    main(Config())
