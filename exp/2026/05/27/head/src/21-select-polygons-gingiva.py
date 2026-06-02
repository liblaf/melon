from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    gingiva: Path = cherries.input("20-gingiva.ply")
    output: Path = cherries.output("21-gingiva.polygons.json")


def main(cfg: Config) -> None:
    gingiva: pv.PolyData = melon.io.load_polydata(cfg.gingiva)
    polygons: Integer[np.ndarray, " N"] = melon.io.load_polygons(cfg.output)
    polygons: Integer[np.ndarray, " N"] = melon.ext.select_polygons(gingiva, polygons)
    melon.io.save_polygons(polygons, cfg.output)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
