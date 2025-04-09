from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    mandible: Path = (
        grapes.find_project_dir()
        / "data/02-intermediate/Skeletons/Mandibula_skull001.ply"
    )
    show: bool = False


def main(cfg: Config) -> None:
    mandible: pv.PolyData = melon.load_poly_data(cfg.mandible)
    pivot_left_id: int = mandible.points[:, 0].argmax()  # pyright: ignore[reportAssignmentType]
    pivot_left: Float[np.ndarray, " 3"] = mandible.points[pivot_left_id]
    pivot_right_id: int = mandible.points[:, 0].argmin()  # pyright: ignore[reportAssignmentType]
    pivot_right: Float[np.ndarray, " 3"] = mandible.points[pivot_right_id]
    ic(pivot_left, pivot_right)
    direction: Float[np.ndarray, " 3"] = pivot_left - pivot_right
    direction /= np.linalg.norm(direction)
    ic(direction)
    if cfg.show:
        pl = pv.Plotter()
        pl.add_mesh(mandible)
        pl.add_point_labels(
            [pivot_left, pivot_right],
            ["Left Pivot", "Right Pivot"],
            point_size=20,
            render_points_as_spheres=True,
            always_visible=True,
        )
        pl.show()


if __name__ == "__main__":
    cherries.run(main)
