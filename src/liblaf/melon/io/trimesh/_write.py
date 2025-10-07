from pathlib import Path

import trimesh as tm

from liblaf.melon.io._save import save


@save.register(tm.Trimesh, [".obj", ".off", ".ply", ".stl"])
def save_trimesh(path: Path, obj: tm.Trimesh, /, **kwargs) -> None:
    obj.export(path, **kwargs)
