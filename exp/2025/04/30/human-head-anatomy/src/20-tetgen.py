from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    cranium: Path = cherries.data("02-intermediate/cranium.vtp")
    mandible: Path = cherries.data("02-intermediate/mandible.vtp")
    skin: Path = cherries.data("02-intermediate/skin-with-mouth-socket.ply")

    output: Path = cherries.data("02-intermediate/20-tetgen.vtu")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.cranium)
    cranium: pv.PolyData = melon.load_poly_data(cfg.cranium)
    cherries.log_input(cfg.mandible)
    mandible: pv.PolyData = melon.load_poly_data(cfg.mandible)
    cherries.log_input(cfg.skin)
    skin: pv.PolyData = melon.load_poly_data(cfg.skin)

    tetmesh: pv.UnstructuredGrid = melon.tetwild(
        {
            "operation": "difference",
            "left": skin,
            "right": {
                "operation": "union",
                "left": cranium,
                "right": mandible,
            },
        },
        lr=1.0 / 70.0,
        epsr=1e-3,
        csg=True,
    )

    melon.save(cfg.output, tetmesh)
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.run(main)
