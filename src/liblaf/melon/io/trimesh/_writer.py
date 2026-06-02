from pathlib import Path

import trimesh as tm
from trimesh.exchange.export import _mesh_exporters

from liblaf.melon.io.abc import save


@save.register(tm.Trimesh, (f".{suffix}" for suffix in _mesh_exporters))
def _save_trimesh(obj: tm.Trimesh, path: Path, /, **kwargs) -> None:
    obj.export(path, **kwargs)
