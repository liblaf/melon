from typing import Any

from . import (
    NearestAlgorithm,
    NearestAlgorithmPrepared,
    NearestPointOnSurface,
    NearestResult,
)


def nearest(
    data: Any, query: Any, algo: NearestAlgorithm | None = None
) -> NearestResult:
    if algo is None:
        algo = NearestPointOnSurface()
    prepared: NearestAlgorithmPrepared = algo.prepare(data)
    return prepared.query(query)
