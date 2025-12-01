from typing import Any

import pyvista as pv

from liblaf.melon import io


def compute_volume(mesh: Any) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    result: pv.UnstructuredGrid = mesh.compute_cell_sizes(
        length=False, area=False, volume=True
    )  # pyright: ignore[reportAssignmentType]
    return result
