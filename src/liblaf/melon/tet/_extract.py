from typing import Any

import numpy as np
import pyvista as pv
from pyvista import VectorLike

from liblaf.melon import io


def extract_cells(
    mesh: Any, ind: int | VectorLike[int] | VectorLike[bool], *, invert: bool = False
) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    ind = np.asarray(ind)
    if np.isdtype(ind.dtype, "bool"):
        ind = np.flatnonzero(ind)
    result: pv.UnstructuredGrid = mesh.extract_cells(ind, invert=invert)  # pyright: ignore[reportAssignmentType]
    return result
