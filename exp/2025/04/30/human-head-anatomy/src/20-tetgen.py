from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("02-intermediate/14-cranium.ply")
    mandible: Path = cherries.input("02-intermediate/14-mandible.ply")
    skin: Path = cherries.input("02-intermediate/12-skin.ply")

    output: Path = cherries.output("02-intermediate/20-tetgen.vtu")

    lr: float = 0.05 * 0.5
    epsr: float = 1e-3 * 0.5


def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_poly_data(cfg.cranium)
    mandible: pv.PolyData = melon.load_poly_data(cfg.mandible)
    skin: pv.PolyData = melon.load_poly_data(cfg.skin)
    skull: pv.PolyData = pv.merge([cranium, mandible])
    tetmesh: pv.UnstructuredGrid = melon.tetwild(
        pv.merge([skull.flip_faces(), skin]), lr=cfg.lr, epsr=cfg.epsr
    )
    cherries.log_metric("n_points", tetmesh.n_points)
    cherries.log_metric("n_cells", tetmesh.n_cells)
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
