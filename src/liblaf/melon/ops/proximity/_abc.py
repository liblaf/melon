import abc
from typing import Any

import attrs
import numpy as np
from jaxtyping import Bool, Float


@attrs.frozen(kw_only=True)
class NearestResult:
    distance: Float[np.ndarray, " N"]
    missing: Bool[np.ndarray, " N"]
    nearest: Float[np.ndarray, "N 3"]

    @property
    def n_points(self) -> int:
        return len(self.distance)


class NearestAlgorithmPrepared(abc.ABC):
    @abc.abstractmethod
    def query(self, query: Any) -> NearestResult: ...


class NearestAlgorithm(abc.ABC):
    @abc.abstractmethod
    def prepare(self, data: Any) -> NearestAlgorithmPrepared: ...
