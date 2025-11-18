from pathlib import Path

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher

load_multi_block: ReaderDispatcher[pv.MultiBlock] = ReaderDispatcher(pv.MultiBlock)


@load_multi_block.register(".vtm")
def load_multi_block_pyvista(path: Path, **kwargs) -> pv.MultiBlock:
    return pv.read(path, **kwargs)  # pyright: ignore[reportReturnType]
