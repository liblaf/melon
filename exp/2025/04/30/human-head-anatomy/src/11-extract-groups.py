from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    groups: Path = cherries.data("02-intermediate/groups.toml")
    input: Path = cherries.data("01-raw/Full human head anatomy.obj")
    output: Path = cherries.data("02-intermediate/")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.input)
    groups: dict[str, list[int]] = grapes.load(cfg.groups)
    for subgroup, names in groups.items():
        mesh: pv.PolyData = melon.triangle.extract_groups(full, names)
        melon.save(cfg.output / f"{subgroup}.ply", mesh)
        for name in names:
            mesh: pv.PolyData = melon.triangle.extract_groups(full, name)
            melon.save(cfg.output / subgroup / f"{name}.ply", mesh)


if __name__ == "__main__":
    cherries.run(main)
