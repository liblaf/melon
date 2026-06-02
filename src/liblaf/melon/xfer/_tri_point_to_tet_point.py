from collections.abc import Iterable
from typing import Any

import numpy as np
import pyvista as pv
import scipy.spatial


def tri_point_to_tet_point(
    source: pv.PolyData,
    target: pv.UnstructuredGrid,
    names: Iterable[str],
    *,
    fill_value: Any = None,
) -> pv.UnstructuredGrid:
    kdtree: scipy.spatial.KDTree = scipy.spatial.KDTree(target.points)
    _d, indices = kdtree.query(source.points)
    for name in names:
        source_data: np.ndarray = source.point_data[name]
        target_data: np.ndarray = np.full(
            (target.n_points, *source_data.shape[1:]), fill_value, source_data.dtype
        )
        target_data[indices] = source_data
        target.point_data[name] = target_data
    return target
