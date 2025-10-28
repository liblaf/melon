from collections.abc import Mapping, Sequence

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.melon.io.abc import ConverterDispatcher

as_pointset: ConverterDispatcher[pv.PointSet] = ConverterDispatcher(pv.PointSet)


@as_pointset.register(Mapping)
def mapping_to_pointset(obj: Mapping, **kwargs) -> pv.PointSet:
    kwargs.pop("point_normals", None)
    points: Float[np.ndarray, "P 3"] = np.asarray(obj["points"])
    return pv.PointSet(points, **kwargs)


@as_pointset.register(Sequence)
@as_pointset.register(np.ndarray)
def numpy_to_pointset(obj: ArrayLike, **kwargs) -> pv.PointSet:
    kwargs.pop("point_normals", None)
    points: Float[np.ndarray, "P 3"] = np.asarray(obj)
    return pv.PointSet(points, **kwargs)


@as_pointset.register(pv.DataSet)
def dataset_to_pointset(obj: pv.DataSet, **kwargs) -> pv.PointSet:
    kwargs.pop("point_normals", None)
    return obj.cast_to_pointset(**kwargs)


@as_pointset.register(pv.PolyData)
def polydata_to_pointset(
    obj: pv.PolyData, *, point_normals: bool = False, **kwargs
) -> pv.PointSet:
    result: pv.PointSet = obj.cast_to_pointset(**kwargs)
    if point_normals:
        result.point_data["Normals"] = obj.point_normals
    return result


@as_pointset.register(tm.Trimesh)
def trimesh_to_pointset(
    obj: tm.Trimesh, *, point_normals: bool = False, **kwargs
) -> pv.PointSet:
    result = pv.PointSet(obj.vertices, **kwargs)
    if point_normals:
        result.point_data["Normals"] = obj.vertex_normals
    return result
