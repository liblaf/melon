from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import save


@save.register(pv.UnstructuredGrid, pv.UnstructuredGrid._WRITERS.keys())  # noqa: SLF001
def _save_unstructured_grid(obj: pv.UnstructuredGrid, path: Path, /, **kwargs) -> None:
    obj.save(path, **kwargs)
