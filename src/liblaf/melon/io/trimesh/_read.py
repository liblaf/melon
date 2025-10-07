from pathlib import Path

import trimesh as tm

from liblaf.melon.io.abc import ReaderDispatcher

load_trimesh: ReaderDispatcher[tm.Trimesh] = ReaderDispatcher(tm.Trimesh)


@load_trimesh.register(".obj", ".off", ".ply", ".stl")
def load_trimesh_trimesh(path: Path, **kwargs) -> tm.Trimesh:
    return tm.load_mesh(path, **kwargs)
