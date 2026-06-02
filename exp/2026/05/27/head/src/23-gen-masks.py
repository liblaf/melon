from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    skin: Path = cherries.input("22-skin.vtp")
    gingiva: Path = cherries.input("20-gingiva.ply")


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.io.load_polydata(cfg.skin)
    gingiva: pv.PolyData = melon.io.load_polydata(cfg.gingiva)
    skin.point_data["_PointId"] = np.arange(skin.n_points)

    edge_length: Float[np.ndarray, " e"] = melon.tri.edge_length(skin)
    edge_length_min: float = edge_length.min()

    skin.compute_implicit_distance(gingiva, inplace=True)
    implicit_distance: Float[np.ndarray, " p"] = skin.point_data["implicit_distance"]
    free_mask: Bool[np.ndarray, " p"] = np.ones((skin.n_points,), np.bool)


if __name__ == "__main__":
    cherries.main(main)
