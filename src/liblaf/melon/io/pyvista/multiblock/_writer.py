from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pyvista as pv

from liblaf.melon.io.abc import save

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


@save.register(pv.MultiBlock, pv.MultiBlock._WRITERS.keys())  # noqa: SLF001
def _save_multiblock(obj: pv.MultiBlock, path: Path, /, **kwargs) -> None:
    obj.save(path, **kwargs)


@save.register(pv.MultiBlock, (".vtkhdf",))
def _save_multiblock_vtkhdf(obj: pv.MultiBlock, path: Path, /, **kwargs) -> None:
    obj.save(path, **kwargs)
    if shutil.which("h5repack"):
        output: Path = path.with_name(path.name + ".repack")
        args: list[StrOrBytesPath] = [
            "h5repack",
            "--low=4",
            "--high=4",
            "--filter=SHUF",
            "--filter=GZIP=4",
            path,
            output,
        ]
        subprocess.run(args, check=True)
        shutil.move(output, path)
