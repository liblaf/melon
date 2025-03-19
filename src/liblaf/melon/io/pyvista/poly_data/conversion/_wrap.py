import pyvista as pv
import trimesh as tm

from liblaf import melon


class WrapToPolyData(melon.io.AbstractConverter):
    type_from = tm.Trimesh
    type_to = pv.PolyData

    def convert(self, obj: tm.Trimesh) -> pv.PolyData:
        return pv.wrap(obj)  # pyright: ignore[reportReturnType]
