from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    blendshapes: Path = cherries.input("61-blendshapes.vtp")


def main(cfg: Config) -> None:
    blendshapes: pv.PolyData = pv.read(cfg.blendshapes, cls=pv.PolyData)
    mesh: pv.UnstructuredGrid = pv.read(
        cherries.input(f"42-tetmesh-{cfg.suffix}.vtu"), cls=pv.UnstructuredGrid
    )
    blendshapes.triangulate(inplace=True)
    expression_names: list[str] = blendshapes.field_data["ExpressionName"].tolist()
    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface: pv.PolyData = melon.xfer.tri_point_to_tri_point(
        blendshapes, surface, names=expression_names, max_dist=0.01
    )
    nan_mask: Bool[np.ndarray, " p"] = np.any(
        np.isnan(surface.point_data[expression_names[0]]), axis=-1
    )
    surface: pv.PolyData = melon.tri.fill_point(
        surface, nan_mask, names=expression_names, limit=0.01
    )
    mesh: pv.UnstructuredGrid = melon.xfer.tri_point_to_tet_point(
        surface, mesh, fill_values=dict.fromkeys(expression_names, 0.0)
    )
    melon.save(mesh, cherries.output(f"62-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
