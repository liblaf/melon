from collections.abc import Mapping
from typing import Any

import numpy as np
import pyvista as pv
import scipy.spatial


def tri_point_to_tet_point(
    source: pv.PolyData, target: pv.UnstructuredGrid, fill_values: Mapping[str, Any]
) -> pv.UnstructuredGrid:
    """Transfer point arrays from a surface to nearest tetrahedral points.

    Args:
        source: Surface with source point-data arrays.
        target: Tetrahedral mesh whose point data should be filled.
        fill_values: Mapping from point-data array name to default value for
            target points that do not receive a nearest source value.

    Returns:
        The mutated `target` mesh.
    """
    kdtree: scipy.spatial.KDTree = scipy.spatial.KDTree(target.points)
    _d, indices = kdtree.query(source.points)
    for name, fill_value in fill_values.items():
        source_data: np.ndarray = source.point_data[name]
        target_data: np.ndarray = np.full(
            (target.n_points, *source_data.shape[1:]), fill_value, source_data.dtype
        )
        target_data[indices] = source_data
        target.point_data[name] = target_data
    return target
