from pathlib import Path

import pyvista as pv

from liblaf.melon.io._save import save


@save.register(pv.UnstructuredGrid, [".vtu"])
def save_unstructured_grid(path: Path, obj: pv.UnstructuredGrid, /, **kwargs) -> None:
    obj.save(path, **kwargs)
