import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher

from ._mapping import MappingToPolyData
from ._wrap import WrapToPolyData

as_poly_data = ConverterDispatcher(pv.PolyData)
as_poly_data.register(MappingToPolyData())
as_poly_data.register(WrapToPolyData())
