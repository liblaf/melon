import os
from typing import Any

from . import abc, melon, pyvista

abc.register_reader(melon.DICOMReader())
abc.register_reader(pyvista.PolyDataReader())
abc.register_reader(pyvista.UnstructuredGridReader())


def load(path: str | os.PathLike[str]) -> Any:
    return abc.load(path)
