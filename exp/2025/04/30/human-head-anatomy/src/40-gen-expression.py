from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    tetgen: Path = cherries.input("02-intermediate/34-tetgen.vtu")
    source: Path = cherries.input("40-expression-flame.vtp")

    output: Path = cherries.output("40-expression.vtu")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(cfg.tetgen)
    source: pv.PolyData = melon.io.load_polydata(cfg.source)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]

    surface = surface.sample(
        source,
        tolerance=0.01 * surface.length,
        snap_to_closest_point=True,
    )  # pyright: ignore[reportAssignmentType]
    ic(surface.point_data["displacement"])
    tetmesh = melon.tetra.transfer_point_data_from_surface(
        surface,
        tetmesh,
        data=["displacement"],
        fill=0.0,
        # nearest=melon.proximity.NearestPoint(normal_threshold=0.0),
    )
    melon.io.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.run(main)
