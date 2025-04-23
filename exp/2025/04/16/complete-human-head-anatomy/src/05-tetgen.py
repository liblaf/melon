from collections.abc import Mapping
from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    edge_length_arc: float = 0.5
    full: Path = grapes.find_project_dir() / "data/01-raw/Full human head anatomy.obj"
    face: Path = (
        grapes.find_project_dir() / "data/03-primary/skin-with-mouth-socket.ply"
    )
    output: Path = grapes.find_project_dir() / "data/03-primary/tetgen.vtu"
    groups: Path = grapes.find_project_dir() / "data/02-intermediate/groups.toml"


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    face: pv.PolyData = melon.load_poly_data(cfg.face)
    groups: Mapping[str, list[str]] = grapes.load(cfg.groups)
    skull: pv.PolyData = melon.triangle.extract_groups(
        full,
        [
            component
            for key in ["cranium", "mandible", "upper-teeth", "lower-teeth"]
            for component in groups[key]
        ],
    )
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([face, skull])
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
