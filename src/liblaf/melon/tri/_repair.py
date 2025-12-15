import pyvista as pv
import trimesh as tm

from liblaf.melon import io


def fix_inversion(mesh: pv.PolyData) -> pv.PolyData:
    # pyvista.PolyData.compute_normals(auto_orient_normals=True) sometimes produces inverted normals
    mesh_tm: tm.Trimesh = io.as_trimesh(mesh)
    mesh_tm = mesh_tm.fix_normals()
    mesh.copy_structure(io.as_polydata(mesh_tm))
    return mesh
