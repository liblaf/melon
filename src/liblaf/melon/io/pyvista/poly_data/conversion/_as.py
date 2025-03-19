from typing import Any

import pyvista as pv

from liblaf.melon.io import conversion_dispatcher

from . import MappingToPolyData, WrapToPolyData

conversion_dispatcher.register(MappingToPolyData())
conversion_dispatcher.register(WrapToPolyData())


def as_poly_data(obj: Any) -> pv.PolyData:
    return conversion_dispatcher.convert(obj, pv.PolyData)
