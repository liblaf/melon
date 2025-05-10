import trimesh as tm

from liblaf.melon.io.abc import ConverterDispatcher

from ._poly_data import PolyDataToTrimesh

as_trimesh = ConverterDispatcher(tm.Trimesh)
as_trimesh.register(PolyDataToTrimesh())
