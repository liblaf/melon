import io
from pathlib import Path

import einops
import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.cherries as cherries  # noqa: PLR0402
import liblaf.melon as melon  # noqa: PLR0402

PICKED_POINTS_RAW: str = """
-16.224176 82.714378 30.822733
-29.040668 83.874290 65.429703
22.384373 83.141403 32.309559
32.414909 85.249756 70.671371
"""


class Config(cherries.BaseConfig):
    radius: float = 10.0
    raw: Path = Path("./data/00-raw/")


@cherries.main()
def main(cfg: Config) -> None:
    points: Float[np.ndarray, "N 3"] = np.loadtxt(io.StringIO(PICKED_POINTS_RAW))
    muscles: list[pv.PolyData] = []
    muscles.append(make_muscle(points[0:2], radius=cfg.radius))
    muscles.append(make_muscle(points[2:4], radius=cfg.radius))
    for muscle_id, muscle in enumerate(muscles):
        melon.save(cfg.raw / "muscles" / f"{muscle_id:02d}.vtp", muscle)


def make_muscle(points: Float[np.ndarray, "2 3"], radius: float) -> pv.PolyData:
    center: Float[np.ndarray, " 3"] = points.mean(axis=0)
    direction: Float[np.ndarray, " 3"] = points[1] - points[0]
    height: float = np.linalg.norm(direction)  # pyright: ignore[reportAssignmentType]
    direction /= height
    muscle: pv.PolyData = pv.Cylinder(  # pyright: ignore[reportAssignmentType]
        center=center, direction=direction, radius=radius, height=height
    )
    muscle.point_data["orientation"] = einops.repeat(
        direction, "D -> P D", P=muscle.n_points
    )
    muscle.cell_data["orientation"] = einops.repeat(
        direction, "D -> C D", C=muscle.n_cells
    )
    muscle.field_data["orientation"] = direction
    return muscle


if __name__ == "__main__":
    main(Config())
