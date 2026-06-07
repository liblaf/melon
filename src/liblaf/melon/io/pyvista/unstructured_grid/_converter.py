from collections.abc import Mapping
from typing import Any, cast

import meshio
import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher
from liblaf.melon.utils import filter_kwargs

as_unstructured_grid: ConverterDispatcher[pv.UnstructuredGrid] = ConverterDispatcher(
    pv.UnstructuredGrid
)
"""Convert supported volume meshes to [`pyvista.UnstructuredGrid`][pyvista.UnstructuredGrid]."""


@as_unstructured_grid.register(meshio.Mesh)
def _wrap_to_unstructured_grid(obj: meshio.Mesh, /, **kwargs) -> pv.UnstructuredGrid:
    kwargs: Mapping[str, Any] = filter_kwargs(pv.wrap, kwargs)
    return cast("pv.UnstructuredGrid", pv.wrap(obj, **kwargs))
