from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def h5repack(file: Path) -> None:
    if not shutil.which("h5repack"):
        return
    output: Path = file.with_name(file.name + ".repack")
    args: list[StrOrBytesPath] = [
        "h5repack",
        "--low=4",
        "--high=4",
        "--filter=SHUF",
        "--filter=GZIP=4",
        file,
        output,
    ]
    subprocess.run(args, check=True)
    shutil.move(output, file)
