from collections.abc import Iterable
from pathlib import Path
from typing import Any, override

import pyvista as pv

from liblaf.melon.io.abc import Writer

from ._convert import as_polydata


class PolyDataWriter(Writer):
    @property
    @override
    def suffixes(self) -> Iterable[str]:
        return [".geo", ".iv", ".obj", ".ply", ".stl", ".vtk", ".vtkhdf", ".vtp"]

    def __call__(self, path: Path, obj: Any, /, **kwargs) -> None:
        obj: pv.PolyData = as_polydata(obj)
        if path.suffix == ".obj":
            obj = self._remove_materials(obj)  # `.obj` writer is buggy with materials
        obj.save(path, **kwargs)

    def _remove_materials(self, obj: pv.PolyData) -> pv.PolyData:
        obj = obj.copy()
        obj.point_data.active_texture_coordinates_name = None
        if "MaterialNames" in obj.field_data:
            del obj.field_data["MaterialNames"]
        return obj
