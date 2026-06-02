from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    muscles: Path = cherries.input("30-muscles.m.vtkhdf")
    smas: Path = cherries.input("32-smas.vtp")
    mesh: Path = cherries.input("40-tetwild-3191k.vtu")

    output: Path = cherries.output("41-tetmesh-3191k.vtu")


def main(cfg: Config) -> None:
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.muscles)
    smas: pv.PolyData = melon.io.load_polydata(cfg.smas)
    mesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(cfg.mesh)

    muscles_combine: pv.UnstructuredGrid = muscles.combine()
    muscles_combine: pv.PolyData = muscles_combine.extract_surface(algorithm=None)
    mesh.cell_data["MuscleFraction"] = melon.tet.volume_fraction(mesh, muscles_combine)
    mesh.cell_data["SmasFraction"] = melon.tet.volume_fraction(mesh, smas)

    melon.save(mesh, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
