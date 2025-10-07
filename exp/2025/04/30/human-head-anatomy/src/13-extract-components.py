from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")

    cranium: Path = cherries.output("02-intermediate/13-cranium.vtp")
    mandible: Path = cherries.output("02-intermediate/13-mandible.vtp")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict = grapes.load(cfg.groups)
    cranium: pv.PolyData = melon.tri.extract_groups(
        full, groups["cranium"] + groups["upper-teeth"]
    )
    mandible: pv.PolyData = melon.tri.extract_groups(
        full, groups["mandible"] + groups["lower-teeth"]
    )
    melon.save(cfg.cranium, cranium)
    melon.save(cfg.mandible, mandible)


if __name__ == "__main__":
    cherries.run(main)
