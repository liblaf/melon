import abc
from typing import Any, TypedDict

import numpy as np
from jaxtyping import Float, Integer


class NearestResult(TypedDict):
    distance: Float[np.ndarray, " Q"]
    nearest: Float[np.ndarray, "Q 3"]
    vertex_id: Integer[np.ndarray, " Q"]


class NearestPrepared(abc.ABC):
    @abc.abstractmethod
    def query(self, query: Any) -> NearestResult: ...


class Nearest(abc.ABC):
    @abc.abstractmethod
    def prepare(self, source: Any) -> NearestPrepared: ...
