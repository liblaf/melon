from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from jaxtyping import Bool, Integer

from liblaf.melon import io


def fix_winding(mesh: Any, *, check: bool = True) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    flip_mask: Bool[np.ndarray, " C"] = mesh.cell_data["Volume"] < 0
    if np.any(flip_mask):
        mesh = flip(mesh, flip_mask)
        mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
        if check:
            assert np.all(mesh.cell_data["Volume"] >= 0)
    return mesh


def flip(mesh: Any, mask: Bool[npt.ArrayLike, " C"]) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mask: Bool[np.ndarray, " C"] = np.asarray(mask)
    if not np.any(mask):
        return mesh
    tetras: Integer[np.ndarray, "C 4"] = mesh.cells_dict[pv.CellType.TETRA]  # pyright: ignore[reportArgumentType]
    # ref: <https://felupe.readthedocs.io/en/latest/felupe/mesh.html#felupe.mesh.flip>
    faces: Integer[np.ndarray, " 3"] = np.asarray([0, 1, 2], np.int32)
    tetras[np.ix_(mask, faces)] = tetras[np.ix_(mask, faces[::-1])]
    result = pv.UnstructuredGrid({pv.CellType.TETRA: tetras}, mesh.points)
    result.copy_attributes(mesh)
    return result
