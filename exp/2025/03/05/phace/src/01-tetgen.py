from pathlib import Path
from typing import Annotated

import pytetwild
import pyvista as pv
import typer

import liblaf.melon as melon  # noqa: PLR0402


def main(
    data_dir: Annotated[Path, typer.Option("-d", "--data")] = Path("./data/00-raw/"),
) -> None:
    face: pv.PolyData = melon.load_poly_data(data_dir / "face.ply")
    cranium: pv.PolyData = melon.load_poly_data(data_dir / "cranium.ply")
    mandible: pv.PolyData = melon.load_poly_data(data_dir / "mandible.ply")
    skull: pv.PolyData = pv.merge([cranium, mandible])
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([face, skull])
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    melon.save("./data/01-intermediate/00-tetgen.vtu", mesh)


if __name__ == "__main__":
    typer.run(main)
