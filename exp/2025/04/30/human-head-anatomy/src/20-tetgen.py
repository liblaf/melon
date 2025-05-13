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
        lr=0.05 * 0.3,
        epsr=1e-3 * 0.3,
        csg=True,
    )
    cherries.log_metric("n_points", tetmesh.n_points)
    cherries.log_metric("n_cells", tetmesh.n_cells)

    melon.save(cfg.output, tetmesh)
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.run(main)
