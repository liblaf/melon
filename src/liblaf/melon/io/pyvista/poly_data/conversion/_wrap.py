from typing import override

import pyvista as pv
import trimesh as tm

from liblaf.melon.io import abc


class WrapToPolyData(abc.AbstractConverter):
    type_from: type = tm.Trimesh
    type_to: type = pv.PolyData

    @override
    def convert(self, obj: tm.Trimesh, /, **kwargs) -> pv.PolyData:
        return pv.wrap(obj)  # pyright: ignore[reportReturnType]
