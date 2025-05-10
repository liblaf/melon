from pathlib import Path

import pydantic
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    class Outputs(pydantic.BaseModel):
        cranium: Path = cherries.data("02-intermediate/cranium.vtp")
        mandible: Path = cherries.data("02-intermediate/mandible.vtp")

    full: Path = cherries.data("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.data("02-intermediate/groups.toml")
    outputs: Outputs = Outputs()


def main(cfg: Config) -> None:
    cherries.log_input(cfg.full)
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    cherries.log_input(cfg.groups)
    groups: dict = grapes.load(cfg.groups)

    cranium: pv.PolyData = melon.triangle.extract_groups(full, groups["cranium"])
    mandible: pv.PolyData = melon.triangle.extract_groups(full, groups["mandible"])

    melon.save(cfg.outputs.cranium, cranium)
    cherries.log_output(cfg.outputs.cranium)
    melon.save(cfg.outputs.mandible, mandible)
    cherries.log_output(cfg.outputs.mandible)


if __name__ == "__main__":
    cherries.run(main)
