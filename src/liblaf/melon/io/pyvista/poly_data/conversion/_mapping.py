from collections.abc import Mapping
from typing import override

import glom
import pyvista as pv

from liblaf.melon.io import abc


class MappingToPolyData(abc.AbstractConverter):
    type_from: type = Mapping
    type_to: type = pv.PolyData

    @override
    def convert(self, obj: Mapping, /, **kwargs) -> pv.PolyData:
        return pv.PolyData.from_regular_faces(
            obj["points"],
            glom.glom(obj, glom.Coalesce("faces", "cells", "triangles", "quads")),
        )
