from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer

from liblaf import melon


def main(
    raw_dir: Annotated[Path, typer.Option("-r", "--raw")] = Path("./data/00-raw/"),
    intermediate_dir: Annotated[Path, typer.Option("-i", "--intermediate")] = Path(
        "./data/01-intermediate/"
    ),
) -> None:
    face: pv.PolyData = melon.load_poly_data(raw_dir / "face.ply")
    face.triangulate(inplace=True)
    cranium: pv.PolyData = melon.load_poly_data(raw_dir / "cranium.ply")
    mandible: pv.PolyData = melon.load_poly_data(raw_dir / "mandible.ply")
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        intermediate_dir / "00-tetgen.vtu"
    )
    mesh.point_data["point-id"] = np.arange(mesh.n_points)
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface.point_data.update(
        melon.ops.transfer_components(
            surface,
            {"face": face, "cranium": cranium, "mandible": mandible},
            proximity=melon.NearestPointOnSurface(normal_threshold=-np.inf),
        )
    )
    melon.save(intermediate_dir / "01-segmented-surface.vtp", surface)
    for component in ["face", "cranium", "mandible"]:
        mesh.point_data[f"is-{component}"] = False  # pyright: ignore[reportArgumentType]
        mesh.point_data[f"is-{component}"][surface.point_data["point-id"]] = (
            surface.point_data[f"is-{component}"]
        )
    melon.save(intermediate_dir / "01-segmented.vtu", mesh)


if __name__ == "__main__":
    typer.run(main)
