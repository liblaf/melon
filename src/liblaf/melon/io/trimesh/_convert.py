import pyvista as pv
import trimesh as tm

from liblaf.melon.io.abc import ConverterDispatcher

as_trimesh: ConverterDispatcher[tm.Trimesh] = ConverterDispatcher(tm.Trimesh)


@as_trimesh.register(pv.PolyData)
def polydata_to_trimesh(obj: pv.PolyData, **kwargs) -> tm.Trimesh:
    obj = obj.triangulate()  # pyright: ignore[reportAssignmentType]
    kwargs.setdefault("process", False)
    return tm.Trimesh(vertices=obj.points, faces=obj.regular_faces, **kwargs)
