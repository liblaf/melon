from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Array, Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries

SUFFIX: str = env.str("SUFFIX", "-3152k")


class Config(cherries.BaseConfig):
    suffix: str = SUFFIX
    smas_inner: Path = cherries.input("22-smas-inner.vtp")
    smas_outer: Path = cherries.input("22-smas-outer.vtp")
    tetmesh: Path = cherries.input(f"40-expression{SUFFIX}.vtu")

    output: Path = cherries.output(f"41-expression{SUFFIX}.vtu")


def main(cfg: Config) -> None:
    smas_inner: pv.PolyData = melon.load_polydata(cfg.smas_inner)
    smas_outer: pv.PolyData = melon.load_polydata(cfg.smas_outer)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    outer_fraction: Float[Array, " cells"] = melon.tet.compute_volume_fraction(
        tetmesh, smas_outer
    )
    inner_fraction: Float[Array, " cells"] = melon.tet.compute_volume_fraction(
        tetmesh, smas_inner
    )
    tetmesh.cell_data["SmasFraction"] = np.clip(
        np.asarray(outer_fraction - inner_fraction), 0.0, 1.0
    )
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
