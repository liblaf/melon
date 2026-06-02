from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    muscles: Path = cherries.input("30-muscles.m.vtkhdf")
    smas: Path = cherries.input("32-smas.vtp")


def main(cfg: Config) -> None:
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.muscles)
    smas: pv.PolyData = melon.io.load_polydata(cfg.smas)
    mesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        cherries.input(f"40-tetwild-{cfg.suffix}.vtu")
    )

    muscles_combine: pv.UnstructuredGrid = muscles.combine()
    muscles_combine: pv.PolyData = muscles_combine.extract_surface(algorithm=None)
    mesh.cell_data["MuscleFraction"] = melon.tet.volume_fraction(mesh, muscles_combine)
    mesh.cell_data["SmasFraction"] = melon.tet.volume_fraction(mesh, smas)

    melon.save(mesh, cherries.output(f"41-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
