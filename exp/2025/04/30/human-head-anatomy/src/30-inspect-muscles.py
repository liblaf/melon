from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("01-raw/groups.toml")

    muscles: Path = cherries.input("01-raw/muscles.vtp")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    muscles: pv.PolyData = melon.tri.extract_groups(full, groups["Muscles"])

    melon.save(cfg.muscles, muscles)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
