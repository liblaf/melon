from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import platformdirs
import pytetwild
import pyvista as pv

from liblaf.melon.tet import fix_winding


def _wraps[**P, T](
    func: Callable[P, T],
) -> Callable[[Callable[..., Any]], Callable[P, T]]:
    del func

    def wrapper(wrapped: Callable[..., Any]) -> Callable[P, T]:
        return wrapped

    return wrapper


memory: joblib.Memory = joblib.Memory(platformdirs.user_cache_path("joblib"))


@_wraps(pytetwild.tetrahedralize_pv)
@memory.cache
def tetwild(*args, **kwargs) -> pv.UnstructuredGrid:
    """Tetrahedralize a surface with TetWild and repair winding.

    Positional and keyword arguments are forwarded to
    [`pytetwild.tetrahedralize_pv`][pytetwild.tetrahedralize_pv]. Results are
    cached with Joblib under the user cache directory.

    Returns:
        Tetrahedral mesh with non-negative cell volumes.
    """
    try:
        result: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(*args, **kwargs)
    finally:
        Path("__tracked_surface.stl").unlink(missing_ok=True)
    result: pv.UnstructuredGrid = fix_winding(result)
    return result
