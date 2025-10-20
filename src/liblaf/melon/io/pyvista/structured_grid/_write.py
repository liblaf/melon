from pathlib import Path

import pyvista as pv

from liblaf.melon.io._save import save


@save.register(pv.StructuredGrid, [".vts"])
def save_structured_grid(path: Path, obj: pv.StructuredGrid, /, **kwargs) -> None:
    obj.save(path, **kwargs)
