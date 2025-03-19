from pathlib import Path

import numpy as np
import pyvista as pv
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
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    muscles_fixed: list[pv.PolyData] = []
    n_muscles: int = 0
    for muscle_name in groups["muscles"]:
        selection: Bool[np.ndarray, " C"] = melon.select_cells_by_group(
            full, muscle_name
        )
        muscle: pv.PolyData = melon.extract_cells(full, selection)
        bodies: pv.MultiBlock = muscle.split_bodies().as_polydata_blocks()
        n_muscles += len(bodies)
        bodies_fixed: list[pv.PolyData] = []
        for body in bodies:
            body_fixed: pv.PolyData = melon.mesh_fix(body)
            assert melon.is_volume(body_fixed)
            bodies_fixed.append(body_fixed)
        muscle_fixed: pv.PolyData = pv.merge(bodies_fixed)
        assert melon.is_volume(muscle_fixed)
        muscles_fixed.append(muscle_fixed)
        melon.save(cfg.output / "muscles" / f"{muscle_name}.ply", muscle_fixed)
    ic(n_muscles)
    melon.save(cfg.output / "muscles.ply", pv.merge(muscles_fixed))


if __name__ == "__main__":
    main(Config())
