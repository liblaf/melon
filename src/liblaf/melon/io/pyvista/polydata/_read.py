from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

load_polydata: ReaderDispatcher[pv.PolyData] = ReaderDispatcher(pv.PolyData)


@load_polydata.register(".ply", ".stl", ".vtp")
def load_polydata_pyvista(path: Path, **kwargs) -> pv.PolyData:
    return pv.read(path, **kwargs)  # pyright: ignore[reportReturnType]


@load_polydata.register(".obj")
def load_polydata_obj(path: Path, **kwargs) -> pv.PolyData:
    from ._read_obj import load_polydata_obj as _load_polydata_obj

    mesh: pv.PolyData = _load_polydata_obj(path, **kwargs)
    if kwargs.pop("clean", False):
        mesh.clean(inplace=True, **kwargs)
    return mesh
