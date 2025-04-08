from typing import Any

from ._abc import NearestAlgorithm, NearestAlgorithmPrepared, NearestResult
from ._nearest_point_on_surface import NearestPointOnSurface


def nearest(
    data: Any, query: Any, algo: NearestAlgorithm | None = None
) -> NearestResult:
    if algo is None:
        algo = NearestPointOnSurface()
    prepared: NearestAlgorithmPrepared = algo.prepare(data)
    return prepared.query(query)
