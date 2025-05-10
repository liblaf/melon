import shutil
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any

import pyvista as pv

from liblaf.melon import io, tetra


def tetwild(
    surface: Any,
    *,
    fix_winding: bool = True,
    lr: float = 0.05,
    epsr: float = 1e-3,
    level: int = 6,
    **kwargs,
) -> pv.UnstructuredGrid:
    if shutil.which("fTetWild"):
        mesh: pv.UnstructuredGrid = _tetwild_exe(
            surface, lr=lr, epsr=epsr, level=level, **kwargs
        )
    else:
        raise NotImplementedError
    if fix_winding:
        mesh = tetra.fix_winding(mesh)
    return mesh


def _tetwild_exe(
    surface: Any, *, lr: float = 0.05, epsr: float = 1e-3, level: int = 6, **kwargs
) -> pv.UnstructuredGrid:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        input_file: Path = tmpdir / "input.ply"
        output: Path = tmpdir / "output.msh"
        io.save(input_file, surface)
        args: list[str] = [
            "fTetWild",
            "--input",
            str(input_file),
            "--output",
            str(output),
            "--lr",
            str(lr),
            "--epsr",
            str(epsr),
            "--level",
            str(level),
        ]
        for k, v in kwargs.items():
            args.extend([f"--{k}", str(v)])
        sp.run(args, check=True)
        return io.load_unstructured_grid(output)
