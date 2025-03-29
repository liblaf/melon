from typing import Any

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import grapes


def tetwild(
    mesh: Any, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    if grapes.has_module("pytetwild"):
        return _pytetwild(mesh, edge_length_fac=edge_length_fac, optimize=optimize)
    return _tetwild_exe(mesh, edge_length_fac=edge_length_fac, optimize=optimize)


def _pytetwild(
    mesh: Any, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    import pytetwild

    mesh = melon.as_poly_data(mesh)
    mesh = pytetwild.tetrahedralize_pv(
        mesh, edge_length_fac=edge_length_fac, optimize=optimize
    )
    return mesh


def _tetwild_exe(
    mesh: Any, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    # TODO: call external `fTetWild` executable
    raise NotImplementedError
