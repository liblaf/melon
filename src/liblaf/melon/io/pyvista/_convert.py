from typing import Any

import pyvista as pv

from liblaf.melon.io.abc import UnsupportedConverterError

from .poly_data import as_poly_data
from .unstructured_grid import as_unstructured_grid


def as_mesh(mesh: Any) -> pv.PolyData | pv.UnstructuredGrid:
    try:
        return as_poly_data(mesh)
    except UnsupportedConverterError:
        pass
    return as_unstructured_grid(mesh)
