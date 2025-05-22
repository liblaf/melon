from typing import Any, overload

from ._abc import NearestAlgorithm, NearestAlgorithmPrepared, NearestResult
from ._nearest_point import NearestPoint, NearestPointResult
from ._nearest_point_on_surface import (
    NearestPointOnSurface,
    NearestPointOnSurfaceResult,
)


@overload
def nearest(
    source: Any, query: Any, algo: NearestPoint | None = None
) -> NearestPointResult: ...
@overload
def nearest(
    source: Any, query: Any, algo: NearestPointOnSurface
) -> NearestPointOnSurfaceResult: ...
@overload
def nearest(source: Any, query: Any, algo: NearestAlgorithm) -> NearestResult: ...
def nearest(
    source: Any, query: Any, algo: NearestAlgorithm | None = None
) -> NearestResult:
    if algo is None:
        algo = NearestPoint()
    prepared: NearestAlgorithmPrepared = algo.prepare(source)
    return prepared.query(query)
