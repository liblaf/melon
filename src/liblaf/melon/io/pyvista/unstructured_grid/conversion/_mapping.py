from collections.abc import Mapping
from typing import override

import glom
import pyvista as pv

from liblaf.melon.io import abc


class MappingToUnstructuredGrid(abc.AbstractConverter):
    type_from: type = Mapping
    type_to: type = pv.UnstructuredGrid

    @override
    def convert(self, obj: Mapping, /, **kwargs) -> pv.UnstructuredGrid:
        return pv.UnstructuredGrid(
            {pv.CellType.TETRA: glom.glom(obj, glom.Coalesce("tetras", "cells"))},
            obj["points"],
        )
