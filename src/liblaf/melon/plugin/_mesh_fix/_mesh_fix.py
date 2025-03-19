import shutil
from typing import Any

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import grapes


def mesh_fix(
    mesh: Any,
    *,
    check: bool = True,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
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
    if check and not melon.is_volume(mesh):
        msg = "Mesh is not a volume"
        raise ValueError(msg)
    return mesh


def _pymeshfix(
    mesh: Any,
    *,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> pv.PolyData:
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
