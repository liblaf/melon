from collections.abc import Mapping
from typing import Any, override

import glom
import numpy as np
import pyvista as pv

from liblaf.melon.io.abc import AbstractConverter


class MappingToPolyData(AbstractConverter):
    precedence: int = -1

    @override
    def match_from(self, data: Any, /) -> bool:
        return isinstance(data, Mapping)

    @override
    def convert(self, data: Mapping, /, **kwargs) -> pv.PolyData:
        result: pv.PolyData = pv.PolyData.from_regular_faces(
            points=np.asarray(data["points"]),
            faces=np.asarray(
                glom.glom(data, glom.Coalesce("triangles", "quads", "faces", "cells"))
            ),
            **kwargs,
        )
        return result
