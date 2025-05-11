import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

from ._obj import ObjReader
from ._pyvista import PolyDataReader

load_poly_data = ReaderDispatcher(pv.PolyData)
load_poly_data.register(ObjReader())
load_poly_data.register(PolyDataReader())
