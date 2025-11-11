import importlib.util
import shutil
from typing import Any

import pyvista as pv

from liblaf.melon import io, tri


def mesh_fix(
    mesh: Any,
    *,
    check: bool = True,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> pv.PolyData:
    result: pv.PolyData
    if importlib.util.find_spec("pymeshfix") is not None:
        result = _pymeshfix(
            mesh,
            verbose=verbose,
            joincomp=joincomp,
            remove_smallest_components=remove_smallest_components,
        )
    elif shutil.which("MeshFix"):
        result = _mesh_fix_exe(mesh, verbose=verbose)
    else:
        raise NotImplementedError
    if check:
        assert tri.is_volume(result)
    mesh: pv.PolyData = io.as_polydata(mesh)
    result.field_data.update(mesh.field_data)
    return result


def _pymeshfix(
    mesh: Any,
    *,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> pv.PolyData:
    import pymeshfix

    mesh: pv.PolyData = io.as_polydata(mesh)
    fix = pymeshfix.MeshFix(mesh)
    fix.repair(
        verbose=verbose,
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
    )
    return fix.mesh


def _mesh_fix_exe(mesh: Any, *, verbose: bool = False) -> pv.PolyData:
    # TODO: call external `MeshFix` executable
    raise NotImplementedError
