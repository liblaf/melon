from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    n_samples: int = 1000

    muscle: Path = cherries.output("10-muscle.vtp")
    tetgen: Path = cherries.output("10-tetgen.vtu")


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box((-1, 1, -0.5, 0.5, 0, 0.2))
    muscle: pv.PolyData = pv.Box((-1, 1, -0.1, 0.1, 0.09, 0.11))
    tetmesh: pv.UnstructuredGrid = melon.tetwild(surface)
    tetmesh.cell_data["MuscleFraction"] = melon.tet.compute_volume_fraction(  # pyright: ignore[reportArgumentType]
        tetmesh, muscle, n_samples=cfg.n_samples
    )
    melon.save(cfg.muscle, muscle)
    melon.save(cfg.tetgen, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
