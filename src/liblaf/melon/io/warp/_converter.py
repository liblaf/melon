from collections.abc import Mapping
from typing import Any

import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Integer

from liblaf.melon.io.abc import ConverterDispatcher
from liblaf.melon.utils import filter_kwargs

as_warp_mesh: ConverterDispatcher[wp.Mesh] = ConverterDispatcher(wp.Mesh)
"""Convert supported triangular meshes to [`warp.Mesh`][warp.Mesh]."""


@as_warp_mesh.register(pv.PolyData)
def _polydata_to_warp_mesh(obj: pv.PolyData, /, **kwargs) -> wp.Mesh:
    kwargs: Mapping[str, Any] = filter_kwargs(wp.Mesh, kwargs)
    points: wp.array = wp.from_numpy(obj.points, wp.vec3f)
    faces: Integer[np.ndarray, "c 3"] = obj.regular_faces
    indices: wp.array = wp.from_numpy(faces.flatten(), wp.int32)
    return wp.Mesh(points, indices)


@as_warp_mesh.register(tm.Trimesh)
def _trimesh_to_warp_mesh(obj: tm.Trimesh, /, **kwargs) -> wp.Mesh:
    kwargs: Mapping[str, Any] = filter_kwargs(wp.Mesh, kwargs)
    points: wp.array = wp.from_numpy(obj.vertices, wp.vec3f)
    faces: Integer[np.ndarray, "c 3"] = obj.faces
    indices: wp.array = wp.from_numpy(faces.flatten(), wp.int32)
    return wp.Mesh(points, indices)
