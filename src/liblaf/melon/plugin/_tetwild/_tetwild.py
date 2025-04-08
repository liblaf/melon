from typing import Any

import pyvista as pv

from liblaf import grapes
from liblaf.melon import io, tetra


def tetwild(
    mesh: Any,
    *,
    edge_length_fac: float = 0.05,
    fix_winding: bool = True,
    optimize: bool = True,
) -> pv.UnstructuredGrid:
    result: pv.UnstructuredGrid
    if grapes.has_module("pytetwild"):
        result = _pytetwild(mesh, edge_length_fac=edge_length_fac, optimize=optimize)
    else:
        result = _tetwild_exe(mesh, edge_length_fac=edge_length_fac, optimize=optimize)
    if fix_winding:
        result = tetra.fix_winding(result)
    return result


def _pytetwild(
    mesh: Any, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    import pytetwild

    mesh = io.as_poly_data(mesh)
    mesh = pytetwild.tetrahedralize_pv(
        mesh, edge_length_fac=edge_length_fac, optimize=optimize
    )
    return mesh


def _tetwild_exe(
    mesh: Any, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    # TODO: call external `fTetWild` executable
    raise NotImplementedError
