from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    edge_length_arc: float = 0.05
    full: Path = grapes.find_project_dir() / "data/01-raw/Full human head anatomy.obj"
    face: Path = (
        grapes.find_project_dir() / "data/03-primary/skin-with-mouth-socket.ply"
    )
    output: Path = grapes.find_project_dir() / "data/03-primary/tetgen.vtu"
    groups: Path = grapes.find_project_dir() / "data/02-intermediate/groups.toml"


def extract_component(full: pv.PolyData, group: str) -> pv.PolyData:
    result: pv.PolyData = melon.triangle.extract_groups(full, group)
    result = melon.plugin.mesh_fix(result)
    return result


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    face: pv.PolyData = melon.load_poly_data(cfg.face)
    groups: Mapping[str, list[str]] = grapes.load(cfg.groups)
    skull_components: list[pv.PolyData] = [
        extract_component(full, component)
        # for subgroup in ["cranium", "mandible"]
        for subgroup in ["cranium", "mandible", "upper-teeth", "lower-teeth"]
        for component in groups[subgroup]
    ]
    skull: pv.PolyData = melon.as_poly_data(
        tm.boolean.union([melon.as_trimesh(c) for c in skull_components])
    )
    melon.save("skull.ply", skull)
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([face, skull])
    edge_lengths: Float[np.ndarray, " E"] = melon.triangle.compute_edge_lengths(surface)
    ic(np.max(edge_lengths) / surface.length)
    ic(np.mean(edge_lengths) / surface.length)
    ic(np.median(edge_lengths) / surface.length)
    ic(np.min(edge_lengths) / surface.length)
    mesh: pv.UnstructuredGrid = melon.tetwild(
        surface, edge_length_fac=cfg.edge_length_arc
    )
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
