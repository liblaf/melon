from pathlib import Path

import numpy as np
import pyvista as pv

import liblaf.cherries as cherries  # noqa: PLR0402
from liblaf import melon


class Config(cherries.BaseConfig):
    raw: Path = Path("./data/00-raw/")
    intermediate: Path = Path("./data/01-intermediate/")


def main(cfg: Config) -> None:
    face: pv.PolyData = melon.load_poly_data(cfg.raw / "face.ply")
    face.triangulate(inplace=True)
    cranium: pv.PolyData = melon.load_poly_data(cfg.raw / "cranium.ply")
    mandible: pv.PolyData = melon.load_poly_data(cfg.raw / "mandible.ply")
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cfg.intermediate / "00-tetgen.vtu"
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
    melon.save(cfg.intermediate / "01-segmented-surface.vtp", surface)
    for component in ["face", "cranium", "mandible"]:
        mesh.point_data[f"is-{component}"] = False  # pyright: ignore[reportArgumentType]
        mesh.point_data[f"is-{component}"][surface.point_data["point-id"]] = (
            surface.point_data[f"is-{component}"]
        )
    melon.save(cfg.intermediate / "01-segmented.vtu", mesh)


if __name__ == "__main__":
    main(Config())
