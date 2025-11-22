import pyvista as pv
import trimesh as tm
import warp as wp

from liblaf.melon.io.abc import ConverterDispatcher

as_warp_mesh: ConverterDispatcher[wp.Mesh] = ConverterDispatcher(wp.Mesh)


@as_warp_mesh.register(pv.PolyData)
def polydata_to_warp_mesh(obj: pv.PolyData, **kwargs) -> wp.Mesh:
    obj = obj.triangulate(inplace=False)  # pyright: ignore[reportAssignmentType]
    points: wp.array = wp.from_numpy(obj.points, dtype=wp.vec3f)
    indices: wp.array = wp.from_numpy(obj.regular_faces, dtype=wp.int32).flatten()
    return wp.Mesh(points, indices, **kwargs)


@as_warp_mesh.register(tm.Trimesh)
def trimesh_to_warp_mesh(obj: tm.Trimesh, **kwargs) -> wp.Mesh:
    points: wp.array = wp.from_numpy(obj.vertices, dtype=wp.vec3f)
    indices: wp.array = wp.from_numpy(obj.faces, dtype=wp.int32).flatten()
    return wp.Mesh(points, indices, **kwargs)
