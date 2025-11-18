from pathlib import Path

import numpy as np
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = cherries.input("40-expression-flame.vtp")
    tetmesh: Path = cherries.input("21-tetmesh.vtu")

    output: Path = cherries.output("40-expression.vtu")


def main(cfg: Config) -> None:
    source: pv.PolyData = melon.load_polydata(cfg.source)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    tetmesh.point_data["PointIds"] = np.arange(tetmesh.n_points)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    source = melon.fast_wrapping(source, surface)
    source.compute_normals(auto_orient_normals=True, inplace=True)
    surface.compute_normals(auto_orient_normals=True, inplace=True)
    data_names: list[str] = [
        name for name in source.point_data if name.startswith("Expression")
    ]
    surface = melon.transfer_tri_point(
        source,
        surface,
        data=data_names,
        fill=0.0,
        nearest=melon.NearestPointOnSurface(normal_threshold=-0.5),
    )
    tetmesh = melon.transfer_tri_point_to_tet(
        surface, tetmesh, data=data_names, fill=0.0, point_id="PointIds"
    )
    del tetmesh.point_data["PointIds"]
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
