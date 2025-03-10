from pathlib import Path
from typing import Annotated

import pyvista as pv
import typer

from liblaf import melon


def main(
    intermediate_dir: Annotated[Path, typer.Option("-i", "--intermediate")] = Path(
        "./data/01-intermediate/"
    ),
) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        intermediate_dir / "00-tetgen.vtu"
    )
    voxelize: pv.UnstructuredGrid = pv.voxelize(mesh, check_surface=False)  # pyright: ignore[reportAssignmentType]
    voxelize.save(intermediate_dir / "02-voxelize.vtu")


if __name__ == "__main__":
    typer.run(main)
