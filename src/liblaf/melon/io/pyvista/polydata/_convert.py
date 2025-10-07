from collections.abc import Mapping

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float, Integer

from liblaf.melon.io.abc import ConverterDispatcher

as_polydata: ConverterDispatcher[pv.PolyData] = ConverterDispatcher(pv.PolyData)


@as_polydata.register(Mapping)
def mapping_to_polydata(obj: Mapping, **kwargs) -> pv.PolyData:
    points: Float[np.ndarray, "P 3"] = np.asarray(obj["points"])
    faces: Integer[np.ndarray, "F 3"] = np.asarray(obj["cells"])
    return pv.PolyData.from_regular_faces(points, faces, **kwargs)


@as_polydata.register(pv.Cell)
def cell_to_polydata(obj: pv.Cell, **kwargs) -> pv.PolyData:
    return obj.cast_to_polydata(**kwargs)


@as_polydata.register(tm.Trimesh)
def trimesh_to_polydata(obj: tm.Trimesh, **kwargs) -> pv.PolyData:
    return pv.wrap(obj, **kwargs)
