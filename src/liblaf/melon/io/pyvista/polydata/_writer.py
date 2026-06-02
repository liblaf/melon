from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import save


@save.register(pv.PolyData, pv.PolyData._WRITERS.keys())  # noqa: SLF001
def _save_polydata(obj: pv.PolyData, path: Path, /, **kwargs) -> None:
    obj.save(path, **kwargs)
