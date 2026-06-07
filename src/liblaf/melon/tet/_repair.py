from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io


def fix_winding(mesh: Any, *, check: bool = True) -> pv.UnstructuredGrid:
    """Flip negatively oriented tetrahedra to positive volume.

    Args:
        mesh: Object convertible to [`pyvista.UnstructuredGrid`][pyvista.UnstructuredGrid].
        check: Assert that all tetrahedra have non-negative volume after repair.

    Returns:
        Repaired tetrahedral mesh.
    """
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mesh: pv.UnstructuredGrid = mesh.compute_cell_sizes(
        length=False, area=False, volume=True
    )
    flip_mask: Bool[np.ndarray, " C"] = mesh.cell_data["Volume"] < 0
    if np.any(flip_mask):
        mesh: pv.UnstructuredGrid = flip(mesh, flip_mask)
        mesh: pv.UnstructuredGrid = mesh.compute_cell_sizes(
            length=False, area=False, volume=True
        )
        if check:
            assert np.all(mesh.cell_data["Volume"] >= 0)
    return mesh


def flip(mesh: Any, mask: Bool[ArrayLike, " C"]) -> pv.UnstructuredGrid:
    """Flip tetrahedral winding for selected cells.

    Args:
        mesh: Object convertible to [`pyvista.UnstructuredGrid`][pyvista.UnstructuredGrid].
        mask: Boolean mask selecting tetrahedra to flip.

    Returns:
        Mesh with selected tetrahedra rewound.
    """
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mask: Bool[np.ndarray, " C"] = np.asarray(mask)
    if not np.any(mask):
        return mesh
    tetras: Integer[np.ndarray, "C 4"] = mesh.cells_dict[pv.CellType.TETRA]  # ty:ignore[invalid-argument-type]
    # ref: <https://felupe.readthedocs.io/en/latest/felupe/mesh.html#felupe.mesh.flip>
    faces: Integer[np.ndarray, " 3"] = np.asarray([0, 1, 2], np.int32)
    tetras[np.ix_(mask, faces)] = tetras[np.ix_(mask, faces[::-1])]
    result: pv.UnstructuredGrid = pv.UnstructuredGrid(
        {pv.CellType.TETRA: tetras}, mesh.points
    )
    result.copy_attributes(mesh)
    return result
