from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")

    cranium: Path = cherries.output("02-intermediate/cranium.vtp")
    mandible: Path = cherries.output("02-intermediate/mandible.vtp")
    scalp_head: Path = cherries.output("02-intermediate/scalp_head.vtp")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict = grapes.load(cfg.groups)

    cranium: pv.PolyData = melon.triangle.extract_groups(full, groups["cranium"])
    mandible: pv.PolyData = melon.triangle.extract_groups(full, groups["mandible"])
    scalp_head: pv.PolyData = melon.triangle.extract_groups(full, "Scalp_Head")

    melon.save(cfg.cranium, cranium)
    melon.save(cfg.mandible, mandible)
    melon.save(cfg.scalp_head, scalp_head)


if __name__ == "__main__":
    cherries.run(main)
