from pathlib import Path

import trimesh as tm
from trimesh.exchange.export import _mesh_exporters

from liblaf.melon.io.abc import save


@save.register(tm.Trimesh, _mesh_exporters.keys())
def _save_trimesh(obj: tm.Trimesh, path: Path, /, **kwargs) -> None:
    obj.export(path, **kwargs)
