from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")

    cranium: Path = cherries.output("02-intermediate/cranium.vtp")
    mandible: Path = cherries.output("02-intermediate/mandible.vtp")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict = grapes.load(cfg.groups)

    cranium: pv.PolyData = melon.triangle.extract_groups(full, groups["cranium"])
    mandible: pv.PolyData = melon.triangle.extract_groups(full, groups["mandible"])

    melon.save(cfg.cranium, cranium)
    melon.save(cfg.mandible, mandible)


if __name__ == "__main__":
    cherries.run(main)
