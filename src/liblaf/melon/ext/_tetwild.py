import logging
import os
import shutil
import subprocess as sp
import tempfile
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any, TypedDict, Unpack

import pyvista as pv

from liblaf import grapes
from liblaf.melon import io, tet

type PathLike = str | os.PathLike[str]

logger: logging.Logger = logging.getLogger(__name__)


class TetwildKwargs(TypedDict, total=False):
    lr: float | None
    epsr: float | None
    level: int | None
    no_color: bool | None
    coarsen: bool | None


def tetwild(
    surface: Any,
    *,
    fix_winding: bool = True,
    lr: float | None = None,
    epsr: float | None = None,
    level: int | None = None,
    color: bool = False,
    coarsen: bool = False,
    csg: bool = False,
    **kwargs,
) -> pv.UnstructuredGrid:
    surface: pv.PolyData = copy_structure(surface)  # pyright: ignore[reportAssignmentType]
    if shutil.which("fTetWild"):
        mesh: pv.UnstructuredGrid = _tetwild_exe(
            surface,
            lr=lr,
            epsr=epsr,
            level=level,
            no_color=not color,
            csg=csg,
            coarsen=coarsen,
            **kwargs,
        )
    else:
        raise NotImplementedError
    if fix_winding:
        mesh = tet.fix_winding(mesh)
    return mesh


def copy_structure(surface: Any) -> str | Mapping[str, Any] | pv.PolyData:
    if isinstance(surface, str):
        return surface
    if isinstance(surface, Mapping):
        return {k: copy_structure(v) for k, v in surface.items()}
    surface: pv.PolyData = io.as_polydata(surface)
    structure: pv.PolyData = pv.PolyData()
    structure.copy_structure(surface)
    return structure


@grapes.memorize
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
        if v is None:
            continue
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
        logger.debug("%s", args)
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
