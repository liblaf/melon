import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher

as_multi_block: ConverterDispatcher[pv.MultiBlock] = ConverterDispatcher(pv.MultiBlock)
