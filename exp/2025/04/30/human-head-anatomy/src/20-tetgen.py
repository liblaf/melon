from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("14-cranium.ply")
    mandible: Path = cherries.input("14-mandible.ply")
    skin: Path = cherries.input("12-skin.vtp")

    output: Path = cherries.output("20-tetgen.vtu")

    lr: float = 0.05 * 0.5
    epsr: float = 1e-3 * 0.5


def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)
    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    skull: pv.PolyData = pv.merge([cranium, mandible])
    tetmesh: pv.UnstructuredGrid = melon.tetwild(
        pv.merge([skull.flip_faces(), skin]), lr=cfg.lr, epsr=cfg.epsr
    )
    tetmesh.cell_data.clear()
    ic(tetmesh.n_points)
    ic(tetmesh.n_cells)
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
