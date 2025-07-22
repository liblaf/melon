from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")
    tetgen: Path = cherries.input("02-intermediate/20-tetgen.vtu")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    muscles: pv.PolyData = melon.tri.extract_groups(full, groups["Muscles"])
    muscles: pv.MultiBlock = muscles.split_bodies().as_polydata_blocks()
    muscles: list[pv.PolyData] = [melon.mesh_fix(muscle) for muscle in muscles]
    muscles_volume: float = sum(muscle.volume for muscle in muscles)
    ic(muscles_volume / tetmesh.volume)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
