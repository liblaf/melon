from typing import Any, override

import pyvista as pv
from numpy.typing import ArrayLike

from liblaf.melon.io import abc


class ArrayToPointSet(abc.AbstractConverter):
    type_to: type = pv.PointSet

    @override
    def match_from(self, obj: Any) -> bool:
        return super().match_from(obj)

    @override
    def convert(self, obj: ArrayLike, /, **kwargs) -> pv.PointSet:
        return pv.wrap(obj)
