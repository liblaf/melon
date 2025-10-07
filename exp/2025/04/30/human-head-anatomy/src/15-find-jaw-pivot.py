from pathlib import Path

import numpy as np
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    mandible: pv.PolyData = melon.tri.extract_groups(full, groups["mandible"])
    left_point_id: int = np.argmin(mandible.points[:, 0])  # pyright: ignore[reportAssignmentType]
    right_point_id: int = np.argmax(mandible.points[:, 0])  # pyright: ignore[reportAssignmentType]
    left_point: np.ndarray = mandible.points[left_point_id]
    right_point: np.ndarray = mandible.points[right_point_id]
    print(left_point, right_point)
    print((left_point + right_point) / 2)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
