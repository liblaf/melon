from . import pyvista as _pyvista
from . import trimesh as _trimesh
from .abc import WriterDispatcher

save = WriterDispatcher()
save.register(_pyvista.PolyDataWriter())
save.register(_pyvista.UnstructuredGridWriter())
save.register(_trimesh.TrimeshWriter())
