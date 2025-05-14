import shutil
from typing import Any

import pyvista as pv

from liblaf import grapes
from liblaf.melon import triangle


def mesh_fix(
    mesh: Any,
    *,
    check: bool = True,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = False,
) -> pv.PolyData:
    if grapes.has_module("pymeshfix"):
        mesh: pv.PolyData = _pymeshfix(
            mesh,
            verbose=verbose,
            joincomp=joincomp,
            remove_smallest_components=remove_smallest_components,
        )
    elif shutil.which("MeshFix"):
        mesh: pv.PolyData = _mesh_fix_exe(mesh, verbose=verbose)
    else:
        raise NotImplementedError
    if check:
        assert triangle.is_volume(mesh)
    return mesh


def _pymeshfix(
    mesh: Any,
    *,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> pv.PolyData:
    with grapes.optional_imports():
        import pymeshfix

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
