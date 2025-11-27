from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    tetmesh: Path = cherries.input("30-muscle-fraction.vtu")
    output: Path = cherries.output("31-muscle-inspect.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    mesh = mesh.extract_cells(mesh.cell_data["MuscleFraction"] > 1e-1)  # pyright: ignore[reportAssignmentType]
    mesh.save(cfg.output)


if __name__ == "__main__":
    cherries.main(main)
