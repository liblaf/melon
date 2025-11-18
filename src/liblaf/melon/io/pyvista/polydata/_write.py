from pathlib import Path

import pyvista as pv

from liblaf.melon.io._save import save


@save.register(pv.PolyData, [".ply", ".stl", ".vtp"])
def save_polydata(path: Path, obj: pv.PolyData, /, **kwargs) -> None:
    obj.save(path, **kwargs)


@save.register(pv.PolyData, [".obj"])
def save_polydata_obj(path: Path, obj: pv.PolyData, /, **kwargs) -> None:
    # `.obj` writer is buggy with materials
    obj = obj.copy()
    obj.field_data.pop("MaterialNames", None)
    obj.field_data.pop("MaterialLibraries", None)
    obj.save(path, **kwargs)
