from pathlib import Path

import pyvista as pv

from liblaf.melon.io._save import save


@save.register(pv.PolyData, [".ply", ".stl", ".vtp"])
def save_polydata(path: Path, obj: pv.PolyData, /, **kwargs) -> None:
    obj.save(path, **kwargs)


@save.register(pv.PolyData, [".obj"])
def save_polydata_obj(path: Path, obj: pv.PolyData, /, **kwargs) -> None:
    obj = obj.copy()
    # `.obj` writer is buggy with materials
    obj.point_data.active_texture_coordinates_name = None
    if "MaterialNames" in obj.field_data:
        del obj.field_data["MaterialNames"]
    obj.save(path, **kwargs)
