import abc
from typing import Any, TypedDict

import numpy as np
from jaxtyping import Bool, Float


class NearestResult(TypedDict):
    distance: Float[np.ndarray, " Q"]
    missing: Bool[np.ndarray, " Q"]
    nearest: Float[np.ndarray, "Q 3"]


class NearestAlgorithmPrepared(abc.ABC):
    @abc.abstractmethod
    def query(self, query: Any) -> NearestResult: ...


class NearestAlgorithm(abc.ABC):
    @abc.abstractmethod
    def prepare(self, source: Any) -> NearestAlgorithmPrepared: ...
