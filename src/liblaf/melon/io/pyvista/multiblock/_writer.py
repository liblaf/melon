from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import save
from liblaf.melon.io.utils import h5repack


@save.register(pv.MultiBlock, pv.MultiBlock._WRITERS.keys())  # noqa: SLF001
def _save_multiblock(obj: pv.MultiBlock, path: Path, /, **kwargs) -> None:
    obj.save(path, **kwargs)


@save.register(pv.MultiBlock, (".vtkhdf",))
def _save_multiblock_vtkhdf(obj: pv.MultiBlock, path: Path, /, **kwargs) -> None:
    obj.save(path, **kwargs)
    h5repack(path)
