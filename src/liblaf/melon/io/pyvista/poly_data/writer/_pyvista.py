from collections.abc import Container
from pathlib import Path
from typing import Any, override

import pyvista as pv

from liblaf.melon.io.abc import AbstractWriter
from liblaf.melon.io.pyvista.poly_data.convert import as_poly_data
from liblaf.melon.typed import PathLike


class PolyDataWriter(AbstractWriter):
    extensions: Container[str] = {".geo", ".iv", ".obj", ".ply", ".stl", ".vtp"}

    @override
    def save(self, path: PathLike, data: Any, /, **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: pv.PolyData = as_poly_data(data)
        if path.suffix == ".obj":
            data = self._remove_materials(data)  # `.obj` writer is buggy with materials
        data.save(path, **kwargs)

    def _remove_materials(self, data: pv.PolyData) -> pv.PolyData:
        data = data.copy()
        data.point_data.active_texture_coordinates_name = None
        if "MaterialNames" in data.field_data:
            del data.field_data["MaterialNames"]
        return data
