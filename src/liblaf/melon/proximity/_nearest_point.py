from typing import Any, override

import attrs
import numpy as np
import pyvista as pv
import scipy
import scipy.spatial
from jaxtyping import Float, Integer

from liblaf.melon import io

from ._abc import Nearest, NearestPrepared, NearestResult


@attrs.frozen(kw_only=True)
class NearestPointPrepared(NearestPrepared):
    kdtree: scipy.spatial.KDTree

    distance_upper_bound: float = np.inf
    workers: int

    @override
    def query(self, query: Any) -> NearestResult:
        query: pv.PointSet = io.as_point_set(query)
        distance: Float[np.ndarray, " Q"]
        index: Integer[np.ndarray, " Q"]
        distance, index = self.kdtree.query(query.points, workers=self.workers)
        return NearestResult(
            distance=distance, nearest=self.kdtree.data[index], vertex_id=index
        )


@attrs.define(kw_only=True)
class NearestPoint(Nearest):
    distance_upper_bound: float = np.inf
    workers: int = -1

    @override
    def prepare(self, source: Any) -> NearestPrepared:
        source: pv.PointSet = io.as_point_set(source)
        return NearestPointPrepared(
            kdtree=scipy.spatial.KDTree(source.points),
            distance_upper_bound=self.distance_upper_bound,
            workers=self.workers,
        )
