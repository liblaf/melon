from typing import cast

import pymeshfix
import pyvista as pv
import trimesh as tm


def meshfix(
    mesh: pv.PolyData,
    *,
    check: bool = True,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
    verbose: bool = False,
) -> pv.PolyData:
    fix: pymeshfix.MeshFix = pymeshfix.MeshFix(mesh, verbose=verbose)
    fix.repair(joincomp=joincomp, remove_smallest_components=remove_smallest_components)
    result_pv: pv.PolyData = fix.mesh
    if check:
        assert not result_pv.is_empty
    if result_pv.is_empty:
        return result_pv
    # result_pv.compute_normals(auto_orient_normals=True, inplace=True)
    # result_pv.compute_normals(auto_orient_normals=True) does not work well
    result_tm: tm.Trimesh = pv.to_trimesh(result_pv)
    result_tm.fix_normals()
    if check:
        assert result_tm.is_volume
    result_pv: pv.PolyData = cast("pv.PolyData", pv.wrap(result_tm))
    result_pv.field_data.update(mesh.field_data)
    return result_pv
