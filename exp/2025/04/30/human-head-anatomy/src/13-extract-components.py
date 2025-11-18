from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")

    cranium: Path = cherries.output("13-cranium.vtp")
    mandible: Path = cherries.output("13-mandible.vtp")
    muscles: Path = cherries.output("13-muscles.vtp")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    full.point_data.clear()
    groups: dict = grapes.load(cfg.groups)
    cranium: pv.PolyData = melon.tri.extract_groups(
        full, groups["Cranium"] + groups["UpperTeeth"]
    )
    mandible: pv.PolyData = melon.tri.extract_groups(
        full, groups["Mandible"] + groups["LowerTeeth"]
    )
    muscles: pv.PolyData = melon.tri.extract_groups(full, groups["Muscles"])
    melon.save(cfg.cranium, cranium)
    melon.save(cfg.mandible, mandible)
    melon.save(cfg.muscles, muscles)


if __name__ == "__main__":
    cherries.main(main)
