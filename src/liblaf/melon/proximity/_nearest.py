from typing import Any

from ._abc import Nearest, NearestPrepared, NearestResult
from ._nearest_point import NearestPoint


def nearest(source: Any, query: Any, algo: Nearest | None = None) -> NearestResult:
    if algo is None:
        algo = NearestPoint()
    prepared: NearestPrepared = algo.prepare(source)
    return prepared.query(query)
