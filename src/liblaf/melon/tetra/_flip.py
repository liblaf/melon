from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from jaxtyping import Bool, Integer


def fix_winding(mesh: Any) -> pv.UnstructuredGrid:
    from liblaf.melon import io

    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    flip_mask: Bool[np.ndarray, " C"] = mesh.cell_data["Volume"] < 0
    if np.any(flip_mask):
        mesh = flip(mesh, flip_mask)
        del mesh.cell_data["Volume"]
    return mesh


def flip(mesh: Any, mask: Bool[npt.ArrayLike, " C"]) -> pv.UnstructuredGrid:
    from liblaf.melon import io

    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mask: Bool[np.ndarray, " C"] = np.asarray(mask)
    tetras: Integer[np.ndarray, "C 4"] = mesh.cells_dict[pv.CellType.TETRA]
    tetras[mask] = tetras[mask][:, [0, 3, 2, 1]]
    result = pv.UnstructuredGrid({pv.CellType.TETRA: tetras}, mesh.points)
    result.copy_attributes(mesh)
    return result
