from pathlib import Path
from typing import Any

from liblaf.melon import io
from liblaf.melon.io.abc import Reader

READERS: dict[str, Reader] = {
    ".obj": io.load_polydata,
    ".ply": io.load_polydata,
    ".stl": io.load_polydata,
    ".vtp": io.load_polydata,
    ".msh": io.load_unstructured_grid,
    ".vtu": io.load_unstructured_grid,
}


def convert(input: Path, output: Path, /) -> None:  # noqa: A002
    reader: Reader = READERS[input.suffix]
    obj: Any = reader(input)
    io.save(output, obj)
