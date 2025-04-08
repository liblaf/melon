from typing import Any

import pyvista as pv

from liblaf.melon.io import abc

from ._mapping import MappingToPolyData
from ._wrap import WrapToPolyData

abc.register_converter(MappingToPolyData())
abc.register_converter(WrapToPolyData())


def as_poly_data(obj: Any) -> pv.PolyData:
    return abc.convert(obj, pv.PolyData)
