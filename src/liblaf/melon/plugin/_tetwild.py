import shutil
import subprocess as sp
import tempfile
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any, TypedDict, Unpack

import pyvista as pv
from loguru import logger

from liblaf import grapes
from liblaf.melon import io, tetra
from liblaf.melon.typed import PathLike


class TetwildKwargs(TypedDict, total=False):
    lr: float
    epsr: float
    level: int


def tetwild(
    surface: Any,
    *,
    fix_winding: bool = True,
    lr: float = 0.05,
    epsr: float = 1e-3,
    level: int = 6,
    csg: bool = False,
    **kwargs,
) -> pv.UnstructuredGrid:
    if shutil.which("fTetWild"):
        mesh: pv.UnstructuredGrid = _tetwild_exe(
            surface, lr=lr, epsr=epsr, level=level, csg=csg, **kwargs
        )
    else:
        raise NotImplementedError
    if fix_winding:
        mesh = tetra.fix_winding(mesh)
    return mesh


def _tetwild_exe(
    surface: Any, *, csg: bool = False, **kwargs: Unpack[TetwildKwargs]
) -> pv.UnstructuredGrid:
    if csg:
        return _tetwild_exe_csg(surface, **kwargs)
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        input_file: Path = tmpdir / "input.ply"
        output: Path = tmpdir / "output.msh"
        io.save(input_file, surface)
        args: list[PathLike] = ["fTetWild", "--input", input_file, "--output", output]
        args.extend(_tetwild_exe_args(**kwargs))
        logger.debug(args)
        sp.run(args, check=True)
        return io.load_unstructured_grid(output)


def _tetwild_exe_args(**kwargs: Unpack[TetwildKwargs]) -> Generator[str]:
    for k, v in kwargs.items():
        key: str = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                yield f"--{key}"
        else:
            yield f"--{key}"
            yield str(v)


def _tetwild_exe_csg(csg: Any, **kwargs: Unpack[TetwildKwargs]) -> pv.UnstructuredGrid:
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        csg: Mapping[str, Any] = _save_csg(tmpdir, csg)  # pyright: ignore[reportAssignmentType]
        csg_file: Path = tmpdir / "csg.json"
        grapes.save(csg_file, csg)
        output: Path = tmpdir / "output.msh"
        args: list[PathLike] = ["fTetWild", "--output", output, "--csg", csg_file]
        args.extend(_tetwild_exe_args(**kwargs))
        logger.debug(args)
        sp.run(args, check=True)
        return io.load_unstructured_grid(output)


def _save_csg(tmpdir: Path, csg: Any) -> str | Mapping[str, Any]:
    if isinstance(csg, Mapping):
        return {
            "operation": csg["operation"],
            "left": _save_csg(tmpdir / "left", csg["left"]),
            "right": _save_csg(tmpdir / "right", csg["right"]),
        }
    io.save(tmpdir / "surface.ply", csg)
    return str(tmpdir / "surface.ply")
