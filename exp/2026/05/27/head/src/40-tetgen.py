import logging
from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    skin: Path = cherries.input("22-skin.vtp")
    cranium: Path = cherries.input("13-cranium.ply")
    mandible: Path = cherries.input("13-mandible.ply")

    edge_length_fac: float = 0.005
    epsilon: float = 1e-4
    coarsen: bool = False


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.io.load_polydata(cfg.skin)
    cranium: pv.PolyData = melon.io.load_polydata(cfg.cranium)
    cranium.flip_faces(inplace=True)
    mandible: pv.PolyData = melon.io.load_polydata(cfg.mandible)
    mandible.flip_faces(inplace=True)
    mesh: pv.PolyData = pv.merge([skin, cranium, mandible])
    output: pv.UnstructuredGrid = melon.ext.tetwild(
        mesh,
        edge_length_fac=cfg.edge_length_fac,
        epsilon=cfg.epsilon,
        coarsen=cfg.coarsen,
    )
    output.field_data["EdgeLengthFac"] = cfg.edge_length_fac
    output.field_data["Epsilon"] = cfg.epsilon
    output.field_data["Coarsen"] = cfg.coarsen
    output_path: Path = cherries.output(f"40-tetwild-{output.n_cells / 1000:.0f}k.vtu")
    melon.save(output, output_path)


if __name__ == "__main__":
    cherries.main(main)
