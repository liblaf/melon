from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    muscle: Path = cherries.data("02-intermediate/10-muscle.vtp")
    tetgen: Path = cherries.data("02-intermediate/10-tetgen.vtu")


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box((-1, 1, -0.5, 0.5, 0, 0.2))
    muscle: pv.PolyData = pv.Box((-1, 1, -0.1, 0.1, 0.09, 0.11))
    tetmesh: pv.UnstructuredGrid = melon.tetwild(surface)
    tetmesh.cell_data["muscle-fraction"] = 0.0
    for cell in tetmesh.cell:
        cell: pv.Cell
        barycentric: Float[np.ndarray, "N 3"] = melon.sample_barycentric_coords(
            (100, 4)
        )
        melon.barycentric_to_points(cell.points, barycentric)
        melon.triangle.contains(muscle, points)


if __name__ == "__main__":
    cherries.run(main)
