from collections.abc import Mapping
from typing import Any

import meshio
import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Float

from liblaf.melon.io.abc import ConverterDispatcher
from liblaf.melon.utils import filter_kwargs

as_trimesh: ConverterDispatcher[tm.Trimesh] = ConverterDispatcher(tm.Trimesh)


@as_trimesh.register(meshio.Mesh)
def _meshio_to_trimesh(obj: meshio.Mesh, /, **kwargs) -> tm.Trimesh:
    kwargs: Mapping[str, Any] = filter_kwargs(tm.Trimesh, kwargs)
    return tm.Trimesh(
        vertices=obj.points, faces=obj.get_cells_type("triangle"), **kwargs
    )


@as_trimesh.register(pv.PolyData)
def _polydata_to_trimesh(obj: pv.PolyData, /, **kwargs) -> tm.Trimesh:
    kwargs: Mapping[str, Any] = filter_kwargs(pv.to_trimesh, kwargs)
    return pv.to_trimesh(obj, **kwargs)


@as_trimesh.register(wp.Mesh)
def _warp_to_trimesh(obj: wp.Mesh, /, **kwargs) -> tm.Trimesh:
    kwargs: Mapping[str, Any] = filter_kwargs(tm.Trimesh, kwargs)
    vertices: Float[np.ndarray, "V 3"] = obj.points.numpy()
    faces: Float[np.ndarray, " F*3"] = obj.indices.numpy()
    faces: Float[np.ndarray, "F 3"] = np.reshape(faces, (-1, 3))
    return tm.Trimesh(vertices, faces, **kwargs)
