from collections.abc import Mapping
from typing import Any, cast

import meshio
import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Float

from liblaf.melon.io.abc import ConverterDispatcher
from liblaf.melon.utils import filter_kwargs

as_polydata: ConverterDispatcher[pv.PolyData] = ConverterDispatcher(pv.PolyData)
"""Convert supported mesh objects to [`pyvista.PolyData`][pyvista.PolyData]."""


@as_polydata.register(meshio.Mesh)
@as_polydata.register(tm.Trimesh)
def _wrap_to_polydata(obj: meshio.Mesh | tm.Trimesh, /, **kwargs) -> pv.PolyData:
    kwargs: Mapping[str, Any] = filter_kwargs(pv.wrap, kwargs)
    return cast("pv.PolyData", pv.wrap(obj, **kwargs))


@as_polydata.register(wp.Mesh)
def _warp_to_polydata(obj: wp.Mesh, /, **kwargs) -> pv.PolyData:
    kwargs: Mapping[str, Any] = filter_kwargs(pv.make_tri_mesh, kwargs)
    points: Float[np.ndarray, "P 3"] = obj.points.numpy()
    faces: Float[np.ndarray, " F*3"] = obj.indices.numpy()
    faces: Float[np.ndarray, "F 3"] = np.reshape(faces, (-1, 3))
    return pv.make_tri_mesh(points, faces, **kwargs)
