from typing import Any

import pyvista as pv

from liblaf.melon.io import abc

from ._mapping import MappingToUnstructuredGrid

abc.register_converter(MappingToUnstructuredGrid())


def as_unstructured_grid(obj: Any) -> pv.UnstructuredGrid:
    return abc.convert(obj, pv.UnstructuredGrid)
