import pyvista as pv

from liblaf.melon.io.abc import ConverterDispatcher

from ._array import ArrayToPointSet
from ._poly_data import PolyDataToPointSet
from ._trimesh import TrimeshToPointSet

as_point_set = ConverterDispatcher(pv.PointSet)
as_point_set.register(ArrayToPointSet())
as_point_set.register(PolyDataToPointSet())
as_point_set.register(TrimeshToPointSet())
