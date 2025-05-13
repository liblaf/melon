from typing import Any, override

import pyvista as pv

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractConverter


class ArrayToPointSet(AbstractConverter):
    @override
    def match_from(self, data: Any, /) -> bool:
        return utils.is_array_like(data)

    @override
    def convert(self, data: Any, /, **kwargs) -> pv.PointSet:
        import numpy as np

        return pv.PointSet(np.asarray(data))
