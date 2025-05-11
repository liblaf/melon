import trimesh as tm

from liblaf.melon.io.abc import ReaderDispatcher

from ._trimesh import TrimeshReader

load_trimesh = ReaderDispatcher(tm.Trimesh)
load_trimesh.register(TrimeshReader())
