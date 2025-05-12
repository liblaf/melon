from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.data("01-raw/groups.toml")

    muscles: Path = cherries.data("01-raw/muscles.vtp")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.full)
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    cherries.log_input(cfg.groups)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    muscles: pv.PolyData = melon.triangle.extract_groups(full, groups["Muscles"])

    melon.save(cfg.muscles, muscles)
    cherries.log_output(cfg.muscles)


if __name__ == "__main__":
    cherries.run(main)
