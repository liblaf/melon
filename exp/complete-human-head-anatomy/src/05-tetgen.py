from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    edge_length_arc: float = 0.5
    face: Path = (
        grapes.find_project_dir() / "data/03-primary/skin-with-mouth-socket.ply"
    )
    output: Path = grapes.find_project_dir() / "data/03-primary/tetgen.vtu"
    skull: Path = grapes.find_project_dir() / "data/02-intermediate/Skeletons.ply"


def main(cfg: Config) -> None:
    face: pv.PolyData = melon.load_poly_data(cfg.face)
    skull: pv.PolyData = melon.load_poly_data(cfg.skull)
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([face, skull])
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
