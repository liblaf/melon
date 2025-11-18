from pathlib import Path

import pyvista as pv

from liblaf.melon.io._save import save


@save.register(pv.MultiBlock, [".vtm"])
def save_multi_block(path: Path, obj: pv.MultiBlock, **kwargs) -> None:
    obj.save(path, **kwargs)
