from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

load_multiblock: ReaderDispatcher[pv.MultiBlock] = ReaderDispatcher(pv.MultiBlock)
"""Load a multi-block PyVista dataset."""


@load_multiblock.register_fallback
def _read(path: Path, /, **kwargs) -> pv.MultiBlock:
    return pv.read(path, **kwargs)
