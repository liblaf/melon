from collections.abc import Container
from pathlib import Path
from typing import Any

import pyvista as pv

from liblaf import melon
from liblaf.melon.typed import PathLike

from . import as_poly_data, save_obj


class PolyDataWriter(melon.io.AbstractWriter):
    extensions: Container[str] = {".obj", ".stl", ".vtp", ".ply"}

    def save(self, path: PathLike, obj: Any) -> None:
        path = Path(path)
        obj: pv.PolyData = as_poly_data(obj)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".obj":
            save_obj(path, obj)
        else:
            obj.save(path)
