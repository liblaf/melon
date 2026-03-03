from pathlib import Path

import pyvista as pv
from environs import env

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries

SUFFIX = env.str("SUFFIX", "-515k")


class Config(cherries.BaseConfig):
    suffix: str = SUFFIX
    tetmesh: Path = cherries.input(f"30-muscle-fraction{SUFFIX}.vtu")
    output: Path = cherries.temp(f"31-muscle-inspect{SUFFIX}.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    mesh = mesh.extract_cells(mesh.cell_data["MuscleFraction"] > 1e-3)  # pyright: ignore[reportAssignmentType]
    mesh.save(cfg.output)


if __name__ == "__main__":
    cherries.main(main)
